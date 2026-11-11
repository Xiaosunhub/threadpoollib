#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


/*
 * Keep these in sync with project(VERSION ...) in CMakeLists.txt.
 *
 * Consumers need a compile-time way to detect the API level: batch_size,
 * wait_until() and the expanded PoolMetrics only exist from 2.1.0 on;
 * scheduling (submit_at / submit_after), parallel_for / parallel_reduce
 * and per-worker local queues arrive with 3.0.0.
 *
 * 4.0.0 is a scheduler rewrite: each worker owns a work-stealing deque, so
 * fork-join backlogs (nested submit / parallel_for / parallel_reduce tails)
 * are balanced across idle workers instead of being pinned to the producer.
 * It also adds future composition (when_all / when_any) and a process-wide
 * default_pool() for fire-and-forget use. The stolen counter in
 * PoolMetrics is new in 4.0.0.
 *
 * 5.0.0 turns the task into an execution context: every Task carries a
 * cooperative cancellation token, and every execution path installs that
 * context (token, worker id, owning pool) in thread-local storage before
 * invoking the callable. On top of that it adds cooperative cancellation
 * (submit_token and friends), structured concurrency (TaskGroup),
 * non-blocking future continuations (then), submit_batch and the
 * failed / token_cancelled counters.
 *
 * 5.0.1 adds a task lifecycle hook (Task::on_abandon): discard paths
 * (clear, stop(IMMEDIATE), handle.cancel) now release TaskGroup
 * bookkeeping and fulfil its futures instead of hanging wait(). Group
 * tasks also see their token through ThreadPool::current_token().
 *
 * 5.1.0 adds lazy deadlines to cancellation domains (cancel_after /
 * cancel_at, no timer thread), deadline-bound submissions
 * (submit_with_timeout / submit_with_deadline and priority variants),
 * TaskGroup::cancel_after / cancel_at, and the PoolMetrics.timed_out
 * counter.
 *
 * 5.1.1 hardens future composition: discarded or rejected when_any
 * waiters still release their ticket, so the returned future never
 * hangs; when_all() short-circuits empty input without occupying a
 * worker.
 */
#define THREADPOOL_VERSION_MAJOR 5
#define THREADPOOL_VERSION_MINOR 1
#define THREADPOOL_VERSION_PATCH 1

#define THREADPOOL_VERSION_STRING "5.1.1"


enum class PoolStopMode
{
    GRACEFUL,   // Finish all remaining queued tasks before stopping
    IMMEDIATE   // Discard tasks that have not started yet
};


enum class RejectPolicy
{
    BLOCK,      // Block submit() when the queue is full
    DISCARD,    // Reject the task immediately when the queue is full

    /*
     * Execute the rejected task directly in the submitting thread.
     *
     * This keeps latency bounded for the producer and guarantees the task
     * eventually runs, but it makes the caller block for the whole task.
     */
    CALLER_RUNS
};


enum class TaskState
{
    WAITING,
    RUNNING,
    COMPLETED,
    CANCELLED
};


/*
 * State and packaged_task of one submission share a single heap block.
 *
 * They used to be two independent allocations. The task state is reached
 * through the aliasing shared_ptr constructor, so pointing at it costs no
 * control block either.
 */
namespace threadpool_detail
{
    template <class T>
    struct TaskBody
    {
        template <class Fn>
        explicit TaskBody(Fn&& fn)
            : task(std::forward<Fn>(fn))
        {
        }

        std::atomic<TaskState> state{
            TaskState::WAITING
        };

        std::packaged_task<T()> task;
    };


    /*
     * Shared flag behind CancellationToken / CancellationSource.
     *
     * A separate heap block is deliberate: a token is copied into every
     * task that belongs to a cancellation domain, and cancelling the
     * source must be visible to all of them at once without any registry
     * or callback list to maintain.
     */
    struct CancellationState
    {
        std::atomic<bool> flag{false};


        /*
         * Deadline expressed in steady_clock ticks since its epoch; 0
         * means "no deadline".
         *
         * A deadline is evaluated lazily by whoever asks is_cancelled().
         * No timer thread, no callbacks and no cleanup are needed: the
         * state is kept alive by the tokens themselves, so a task that
         * starts after its deadline simply observes it.
         */
        std::atomic<
            std::chrono::steady_clock::rep
        > deadline{0};
    };


    /*
     * Returns true only when a deadline exists and has passed.
     *
     * The zero check happens before reading the clock, so domains with no
     * deadline never pay for a steady_clock::now() call.
     */
    inline bool deadline_reached(
        const CancellationState& state
    )
    {
        const std::chrono::steady_clock::rep ticks =
            state.deadline.load(
                std::memory_order_acquire
            );


        if(ticks == 0)
            return false;


        return std::chrono::steady_clock::now()
            .time_since_epoch()
            .count()
            >= ticks;
    }


    /*
     * Result type of then(): a void predecessor passes no argument to the
     * continuation, a value predecessor passes its result.
     *
     * The specialization keeps std::invoke_result_t<F, void> from ever
     * being formed - that instantiation is ill-formed for a void-only
     * callable and would break the whole overload.
     */
    template <class T, class F, bool = std::is_void_v<T>>
    struct ThenResult;

    template <class T, class F>
    struct ThenResult<T, F, false>
    {
        using type =
            std::invoke_result_t<F, T>;
    };

    template <class T, class F>
    struct ThenResult<T, F, true>
    {
        using type =
            std::invoke_result_t<F>;
    };
}


/**
 * Thrown when a task observes a cancellation request.
 *
 * A task submitted through submit_token() receives this exception in its
 * future if the token was cancelled before the task started. Long-running
 * tasks can also receive it by calling token.throw_if_cancelled() (or
 * ThreadPool::current_token().throw_if_cancelled()) themselves.
 */
class TaskCancelled : public std::runtime_error
{
public:
    TaskCancelled()
        : std::runtime_error(
            "ThreadPool: task cancelled"
        )
    {
    }

    explicit TaskCancelled(
        const char* what
    )
        : std::runtime_error(what)
    {
    }
};


/**
 * A cheap, copyable view of a cancellation flag.
 *
 * A default-constructed token is never cancellable: cancellable() is
 * false and every check is a single pointer test. Tokens are obtained
 * from CancellationSource::token() or TaskGroup::token().
 *
 * A cancellation domain can also carry a deadline (5.1.0). Deadlines are
 * evaluated lazily at each check, so passing time cancels the domain
 * without any timer thread; timed_out() tells a deadline apart from an
 * explicit request_cancel().
 *
 * This is cooperative cancellation, not thread termination: a task that
 * never checks its token will run to completion even after a request or
 * a deadline.
 */
class CancellationToken
{
public:
    CancellationToken() = default;


    bool is_cancelled() const
    {
        return state_
            &&
            (state_->flag.load(
                std::memory_order_acquire
            )
                ||
                threadpool_detail::deadline_reached(
                    *state_
                ));
    }


    /*
     * True only when the domain was cancelled by its deadline rather than
     * by an explicit request_cancel().
     */
    bool timed_out() const
    {
        return state_
            &&
            !state_->flag.load(
                std::memory_order_acquire
            )
            &&
            threadpool_detail::deadline_reached(
                *state_
            );
    }


    /*
     * Returns the domain's deadline, or a default-constructed time_point
     * when there is none.
     */
    std::chrono::steady_clock::time_point deadline() const
    {
        if(!state_)
            return std::chrono::steady_clock::time_point{};


        const std::chrono::steady_clock::rep ticks =
            state_->deadline.load(
                std::memory_order_acquire
            );


        if(ticks == 0)
            return std::chrono::steady_clock::time_point{};


        return std::chrono::steady_clock::time_point{
            std::chrono::steady_clock::duration{ticks}
        };
    }


    void throw_if_cancelled() const
    {
        if(!is_cancelled())
            return;


        if(timed_out())
        {
            throw TaskCancelled(
                "ThreadPool: task timed out"
            );
        }


        throw TaskCancelled();
    }


    bool cancellable() const
    {
        return static_cast<bool>(state_);
    }


    explicit operator bool() const
    {
        return cancellable();
    }


private:
    explicit CancellationToken(
        std::shared_ptr<
            threadpool_detail::CancellationState
        > state
    )
        : state_(std::move(state))
    {
    }


private:
    std::shared_ptr<
        threadpool_detail::CancellationState
    > state_;


    friend class CancellationSource;
};


/**
 * The writable side of a cancellation domain.
 *
 * request_cancel() is idempotent and thread-safe; it returns true only
 * for the caller that actually flipped the flag from false to true.
 *
 * Since 5.1.0 a domain can also be given a deadline with cancel_after()
 * or cancel_at(). No timer thread is involved: is_cancelled() compares
 * the clock lazily, so a task that has not started yet sees the expiry
 * the moment its wrapper checks the token.
 */
class CancellationSource
{
public:
    CancellationSource()
        : state_(
            std::make_shared<
                threadpool_detail::CancellationState
            >()
        )
    {
    }


    CancellationToken token() const
    {
        return CancellationToken(state_);
    }


    bool request_cancel()
    {
        bool expected = false;

        return state_->flag.compare_exchange_strong(
            expected,
            true,
            std::memory_order_acq_rel
        );
    }


    /*
     * Arm a deadline relative to now. Repeated calls keep the earliest
     * deadline, which is the only update that cannot weaken a guarantee
     * already promised to a token holder.
     */
    template <class Rep, class Period>
    void cancel_after(
        const std::chrono::duration<Rep, Period>& delay
    )
    {
        const auto ticks =
            std::chrono::duration_cast<
                std::chrono::steady_clock::duration
            >(delay).count();


        set_deadline_ticks(
            std::chrono::steady_clock::now()
                .time_since_epoch()
                .count()
            + ticks
        );
    }


    void cancel_at(
        std::chrono::steady_clock::time_point at
    )
    {
        set_deadline_ticks(
            at.time_since_epoch().count()
        );
    }


    bool is_cancelled() const
    {
        return state_->flag.load(
                std::memory_order_acquire
            )
            ||
            threadpool_detail::deadline_reached(
                *state_
            );
    }


    /*
     * True only when a deadline, rather than request_cancel(), cancelled
     * this domain.
     */
    bool timed_out() const
    {
        return !state_->flag.load(
                std::memory_order_acquire
            )
            &&
            threadpool_detail::deadline_reached(
                *state_
            );
    }


    void throw_if_cancelled() const
    {
        if(!is_cancelled())
            return;


        if(timed_out())
        {
            throw TaskCancelled(
                "ThreadPool: task timed out"
            );
        }


        throw TaskCancelled();
    }


private:
    /*
     * Compare-and-swap loop: an existing earlier deadline always wins.
     */
    void set_deadline_ticks(
        std::chrono::steady_clock::rep ticks
    )
    {
        /*
         * Zero is reserved for "no deadline". A non-positive deadline
         * means "already expired", so clamp it to the first tick of the
         * epoch instead of accidentally disarming the domain.
         */
        if(ticks < 1)
            ticks = 1;


        std::chrono::steady_clock::rep current =
            state_->deadline.load(
                std::memory_order_relaxed
            );


        while((current == 0
                ||
                ticks < current)
            &&
            !state_->deadline.compare_exchange_weak(
                current,
                ticks,
                std::memory_order_acq_rel,
                std::memory_order_relaxed
            ))
        {
        }
    }


private:
    std::shared_ptr<
        threadpool_detail::CancellationState
    > state_;
};


/**
 * Task cancellation control handle.
 *
 * Note:
 * cancel() can only cancel a task that has not started running yet.
 * A task that is already RUNNING will not be forcibly terminated.
 */
class TaskHandle
{
public:
    TaskHandle() = default;

    bool cancel()
    {
        if(!state_)
            return false;


        TaskState expected = TaskState::WAITING;

        return state_->compare_exchange_strong(
            expected,
            TaskState::CANCELLED,
            std::memory_order_acq_rel
        );
    }

    bool is_cancelled() const
    {
        if(!state_)
            return false;


        return state_->load(std::memory_order_acquire)
            == TaskState::CANCELLED;
    }

    TaskState state() const
    {
        if(!state_)
            return TaskState::WAITING;


        return state_->load(std::memory_order_acquire);
    }

    bool valid() const
    {
        return static_cast<bool>(state_);
    }

private:
    explicit TaskHandle(
        std::shared_ptr<std::atomic<TaskState>> state)
        : state_(std::move(state))
    {
    }

private:
    /*
     * A default-constructed handle is not attached to any task: it owns no
     * state, and cancel() / is_cancelled() / state() are no-ops on it.
     *
     * Previously this allocated a fresh atomic, which made valid() report
     * true for a handle that could never control anything.
     */
    std::shared_ptr<std::atomic<TaskState>> state_;

    friend class ThreadPool;
};


template <class T>
struct TaskResult
{
    std::future<T> future;
    TaskHandle handle;
};


struct PoolMetrics
{
    std::size_t worker_count = 0;
    std::size_t idle_count = 0;
    std::size_t queue_size = 0;

    std::size_t min_threads = 0;
    std::size_t max_threads = 0;

    /*
     * active = queue_size + tasks already picked up by a worker but not
     * finished yet.
     */
    std::size_t active = 0;

    /*
     * Diagnostics for capacity tuning.
     */
    std::size_t peak_queue = 0;

    std::uint64_t workers_created = 0;

    /*
     * Worker entries currently held, including threads that exited but
     * have not been joined yet.
     *
     * slots == worker_count means every exited worker was reclaimed.
     */
    std::size_t slots = 0;

    std::uint64_t submitted = 0;
    std::uint64_t completed = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t rejected = 0;
    std::uint64_t running = 0;

    /*
     * Tasks taken by an idle worker from a peer's deque.
     *
     * A non-zero value shows the 4.0.0 work-stealing scheduler is actually
     * load-balancing fork-join work rather than leaving it on the producer.
     */
    std::uint64_t stolen = 0;

    /*
     * 5.0.0 diagnostics.
     *
     * failed counts user callables that threw; the exception is still
     * delivered through the task's future as before. A task cancelled
     * through a CancellationToken is not a failure - it is counted by
     * token_cancelled instead - because cancellation is a requested
     * outcome, not an error.
     */
    std::uint64_t failed = 0;

    std::uint64_t token_cancelled = 0;


    /*
     * 5.1.0 diagnostic: tasks that were already expired by their
     * cancellation deadline when the worker picked them up. A subset of
     * token_cancelled, not an additional total.
     */
    std::uint64_t timed_out = 0;


    /*
     * Tasks currently held for a future execution time, i.e. deferred by
     * submit_after() / submit_at(). They are not yet in the runnable queue.
     */
    std::size_t scheduled = 0;
};


class TaskGroup;


class ThreadPool
{
public:

    struct Config
    {
        std::size_t min_threads = 2;
        std::size_t max_threads = 8;

        std::size_t max_queue = 256;

        std::chrono::milliseconds idle_timeout{2000};

        /*
         * Number of tasks a worker takes out of the queue per critical
         * section.
         *
         * 1 is the classic behavior: one pop per task.
         * A larger value amortizes the mutex cost over several tiny tasks.
         *
         * Values below 1 are treated as 1.
         */
        std::size_t batch_size = 1;

        RejectPolicy reject = RejectPolicy::BLOCK;

        /*
         * Optional hook executed once at the beginning of every worker
         * thread, outside of any internal lock.
         *
         * Typical uses:
         *
         * - set a thread name for debugging
         * - initialize thread-local resources
         *
         * Exceptions thrown by the hook are swallowed so that the pool
         * keeps working with a partially initialized worker.
         */
        std::function<void()> thread_init;
    };


public:

    explicit ThreadPool(Config cfg = Config{});

    ~ThreadPool();


    /*
     * A ThreadPool owns threads and tasks.
     *
     * Copying it would duplicate every member including running threads,
     * so copying and moving are disabled.
     */
    ThreadPool(const ThreadPool&) = delete;

    ThreadPool& operator=(const ThreadPool&) = delete;

    ThreadPool(ThreadPool&&) = delete;

    ThreadPool& operator=(ThreadPool&&) = delete;


    /**
     * Submit a normal task.
     *
     * Default priority = 0.
     *
     * Throws std::runtime_error if the pool is stopped or if the task is
     * rejected.
     */
    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<
            std::invoke_result_t<F, Args...>
        >;


    /**
     * Submit a task with priority.
     *
     * A larger priority value means higher execution priority.
     */
    template <class F, class... Args>
    auto submit_priority(
        int priority,
        F&& f,
        Args&&... args
    )
        -> std::future<
            std::invoke_result_t<F, Args...>
        >;


    /**
     * Submit a cancellable task.
     */
    template <class F, class... Args>
    auto submit_cancellable(
        int priority,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    /**
     * Non-throwing submission.
     *
     * Unlike submit(), these never wait for a free queue slot and never
     * throw when the task cannot be accepted.
     *
     * std::nullopt means the queue is full or the pool is stopped.
     */
    template <class F, class... Args>
    auto try_submit(F&& f, Args&&... args)
        -> std::optional<
            std::future<
                std::invoke_result_t<F, Args...>
            >
        >;


    template <class F, class... Args>
    auto try_submit_priority(
        int priority,
        F&& f,
        Args&&... args
    )
        -> std::optional<
            std::future<
                std::invoke_result_t<F, Args...>
            >
        >;


    template <class F, class... Args>
    auto try_submit_cancellable(
        int priority,
        F&& f,
        Args&&... args
    )
        -> std::optional<
            TaskResult<
                std::invoke_result_t<F, Args...>
            >
        >;


    // ============================================================
    // Scheduled tasks (3.0.0)
    // ============================================================

    /**
     * Submit a task that runs after the given delay.
     *
     * Returns a future (priority 0). Throws if the pool is already stopped.
     */
    template <class Rep, class Period, class F, class... Args>
    auto submit_after(
        const std::chrono::duration<
            Rep,
            Period
        >& delay,
        F&& f,
        Args&&... args
    )
        -> std::future<
            std::invoke_result_t<F, Args...>
        >;


    /**
     * Submit a task that runs at the given steady_clock deadline.
     *
     * Priority defaults to 0.
     */
    template <class F, class... Args>
    auto submit_at(
        std::chrono::steady_clock::time_point at,
        F&& f,
        Args&&... args
    )
        -> std::future<
            std::invoke_result_t<F, Args...>
        >;


    template <class F, class... Args>
    auto submit_cancellable_at(
        std::chrono::steady_clock::time_point at,
        int priority,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    template <class Rep, class Period, class F, class... Args>
    auto submit_cancellable_after(
        const std::chrono::duration<
            Rep,
            Period
        >& delay,
        int priority,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    template <class F, class... Args>
    auto try_submit_at(
        std::chrono::steady_clock::time_point at,
        int priority,
        F&& f,
        Args&&... args
    )
        -> std::optional<
            std::future<
                std::invoke_result_t<F, Args...>
            >
        >;


    template <class Rep, class Period, class F, class... Args>
    auto try_submit_after(
        const std::chrono::duration<
            Rep,
            Period
        >& delay,
        int priority,
        F&& f,
        Args&&... args
    )
        -> std::optional<
            std::future<
                std::invoke_result_t<F, Args...>
            >
        >;


    template <class F, class... Args>
    auto try_submit_cancellable_at(
        std::chrono::steady_clock::time_point at,
        int priority,
        F&& f,
        Args&&... args
    )
        -> std::optional<
            TaskResult<
                std::invoke_result_t<F, Args...>
            >
        >;


    template <class Rep, class Period, class F, class... Args>
    auto try_submit_cancellable_after(
        const std::chrono::duration<
            Rep,
            Period
        >& delay,
        int priority,
        F&& f,
        Args&&... args
    )
        -> std::optional<
            TaskResult<
                std::invoke_result_t<F, Args...>
            >
        >;


    // ============================================================
    // Parallel algorithms (3.0.0)
    // ============================================================

    /**
     * Run f(i) for every i in [begin, end) across the pool.
     *
     * The range is split into chunks of at most grainsize indices; each
     * chunk is submitted as a task. Returns a future that completes once
     * every invocation has finished.
     */
    template <class F>
    std::future<void> parallel_for(
        std::size_t begin,
        std::size_t end,
        std::size_t grainsize,
        F&& f
    );


    template <class F>
    std::future<void> parallel_for(
        std::size_t begin,
        std::size_t end,
        F&& f
    );


    /**
     * Parallel reduction.
     *
     * f(begin, end) computes the partial result of one chunk; op(left, right)
     * combines two partial results. init is the neutral element supplied for
     * the whole range. Returns a future with the combined result.
     */
    template <class T, class F, class BinOp>
    std::future<T> parallel_reduce(
        std::size_t begin,
        std::size_t end,
        std::size_t grainsize,
        T init,
        F&& f,
        BinOp&& op
    );


    /**
     * Wait until the queue is empty and every task has finished.
     *
     * NOTE:
     * Never call wait() / wait_for() from inside a task of the same pool.
     * The waiting count includes the caller's own task, so the wait could
     * never complete.
     *
     * Such a call is detected and throws std::logic_error.
     */
    void wait();


    /**
     * Wait with a timeout.
     *
     * Returns true if the pool drained, false on timeout.
     */
    template <class Rep, class Period>
    bool wait_for(
        const std::chrono::duration<
            Rep,
            Period
        >& timeout
    );


    /**
     * Wait until an absolute deadline.
     *
     * Returns true if the pool drained, false on timeout.
     */
    template <class Clock, class Duration>
    bool wait_until(
        const std::chrono::time_point<
            Clock,
            Duration
        >& deadline
    );


    /**
     * Suspend task execution.
     *
     * Already running tasks are not interrupted, but no queued task is
     * picked up until resume() is called.
     */
    void pause();


    void resume();


    bool is_paused() const;


    /**
     * Drop every queued task that has not started yet.
     *
     * The associated TaskHandles are marked CANCELLED.
     *
     * Returns how many tasks were dropped.
     */
    std::size_t clear();


    void set_reject_callback(std::function<void()> cb);


    void stop(
        PoolStopMode mode = PoolStopMode::GRACEFUL
    );


    PoolMetrics get_metrics() const;


    Config get_config() const;


    bool is_stopped() const;


    /**
     * Change the dynamic sizing bounds at runtime.
     *
     * The same rules as the constructor apply:
     *
     *   - min_threads must stay >= 1
     *   - max_threads must stay >= min_threads
     *
     * std::invalid_argument is thrown otherwise.
     *
     * Raising min_threads starts the missing workers immediately, so the
     * new floor is honored without waiting for load.
     *
     * Lowering max_threads never kills a worker that is currently running
     * a task. Surplus workers are shed as they go idle, which is exactly
     * how the pool already settles after a burst.
     */
    void resize(
        std::size_t min_threads,
        std::size_t max_threads
    );


    /**
     * Change the idle timeout at runtime.
     *
     * A worker that is already inside its idle wait_for() picks up the new
     * value on its next wait cycle, so the change takes effect without
     * restarting the pool.
     *
     * Throws std::invalid_argument if timeout <= 0.
     */
    void set_idle_timeout(
        std::chrono::milliseconds timeout
    );


    /**
     * Change the rejection policy at runtime.
     *
     * Applies to the next submit() / try_submit() that finds a full queue,
     * letting a stall be rerouted to DISCARD or CALLER_RUNS on the fly.
     */
    void set_reject_policy(
        RejectPolicy policy
    );


    /**
     * Change the task batch size at runtime.
     *
     * Controls how many tasks a worker pulls out of the queue per critical
     * section. A larger value amortizes mutex cost over many tiny tasks at
     * the expense of poorer priority/fairness; smaller keeps latency low.
     *
     * Values below 1 are treated as 1. Takes effect on the next wakeup,
     * so the pool need not be restarted to retune throughput.
     */
    void set_batch_size(
        std::size_t batch_size
    );


    /**
     * Number of runnable tasks waiting in the global queue (excludes
     * per-worker local queues and not-yet-due scheduled tasks).
     */
    std::size_t queue_size() const;


    /**
     * Number of scheduled tasks that are enqueued but not yet due.
     */
    std::size_t scheduled_count() const;


    /**
     * Number of live worker threads currently owned by the pool.
     */
    std::size_t worker_count() const;


    /**
     * Number of tasks either running on a worker or sitting in a queue
     * (global + local + scheduled) that have not completed yet.
     */
    std::size_t active_count() const;


    // ============================================================
    // Task execution context (5.0.0)
    // ============================================================

    /*
     * Sentinel returned by current_worker_id() on a thread that is not a
     * worker of any ThreadPool.
     */
    static constexpr std::size_t no_worker =
        static_cast<std::size_t>(-1);


    /**
     * Stable id of the worker thread running the current task, or
     * no_worker outside of a task.
     *
     * Ids are process-wide and monotonically assigned, so two pools never
     * report the same id. Useful for thread-local sharding / diagnostics.
     */
    static std::size_t current_worker_id();


    /** The pool owning the current worker thread, or nullptr. */
    static const ThreadPool* current_pool();


    /**
     * Cancellation token attached to the task currently executing.
     *
     * Every execution path (global queue, local deque, stolen task,
     * promoted scheduled task, caller-runs) installs this in thread-local
     * storage before invoking the callable, so a task can cooperatively
     * check cancellation without receiving a token as an argument.
     *
     * Returns a non-cancellable (empty) token for a task submitted without
     * one, and for code outside a task.
     */
    static CancellationToken current_token();


    // ============================================================
    // Cooperative cancellation (5.0.0)
    // ============================================================

    /**
     * Submit a task attached to a cancellation domain.
     *
     * If the token is cancelled before the task starts, the callable is
     * skipped and its future completes with TaskCancelled instead of
     * hanging on a broken promise. A task already running can observe the
     * same token via ThreadPool::current_token().
     */
    template <class F, class... Args>
    auto submit_token(
        CancellationToken token,
        F&& f,
        Args&&... args
    )
        -> std::future<
            std::invoke_result_t<F, Args...>
        >;


    template <class F, class... Args>
    auto submit_token_priority(
        int priority,
        CancellationToken token,
        F&& f,
        Args&&... args
    )
        -> std::future<
            std::invoke_result_t<F, Args...>
        >;


    /**
     * Same as submit_token_priority(), but also returns a TaskHandle so
     * the task can be cancelled individually through the 2.x API.
     */
    template <class F, class... Args>
    auto submit_cancellable_token(
        int priority,
        CancellationToken token,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    template <class F, class... Args>
    auto try_submit_token(
        CancellationToken token,
        F&& f,
        Args&&... args
    )
        -> std::optional<
            std::future<
                std::invoke_result_t<F, Args...>
            >
        >;


    template <class F, class... Args>
    auto submit_at_token(
        std::chrono::steady_clock::time_point at,
        CancellationToken token,
        F&& f,
        Args&&... args
    )
        -> std::future<
            std::invoke_result_t<F, Args...>
        >;


    template <class Rep, class Period, class F, class... Args>
    auto submit_after_token(
        const std::chrono::duration<Rep, Period>& delay,
        CancellationToken token,
        F&& f,
        Args&&... args
    )
        -> std::future<
            std::invoke_result_t<F, Args...>
        >;


    // ============================================================
    // Deadline-bound submissions (5.1.0)
    // ============================================================

    /**
     * Submit a task with its own cancellation deadline.
     *
     * The pool creates a private CancellationSource, arms it with the
     * given deadline and submits through submit_cancellable_token(). If
     * the deadline has passed before the task starts, the callable is
     * skipped and the future throws TaskCancelled (with a "timed out"
     * message). A running task can observe the same deadline through
     * ThreadPool::current_token().throw_if_cancelled().
     *
     * Returns a TaskResult, so the caller still gets a TaskHandle for an
     * individual early cancel() on top of the automatic deadline.
     *
     * This is deliberately lazy: no timer thread is created and nothing
     * is woken at expiry. A task that never checks its token simply runs
     * to completion, exactly like any other cooperative cancellation.
     */
    template <class Rep, class Period, class F, class... Args>
    auto submit_with_timeout(
        const std::chrono::duration<Rep, Period>& timeout,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    template <class F, class... Args>
    auto submit_with_deadline(
        std::chrono::steady_clock::time_point deadline,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    template <class Rep, class Period, class F, class... Args>
    auto submit_with_timeout_priority(
        int priority,
        const std::chrono::duration<Rep, Period>& timeout,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    template <class F, class... Args>
    auto submit_with_deadline_priority(
        int priority,
        std::chrono::steady_clock::time_point deadline,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    // ============================================================
    // Future continuation + batch submission (5.0.0)
    // ============================================================

    /**
     * Chain a callable onto a future without blocking any thread.
     *
     * The continuation is submitted as a pool task; it runs once the
     * predecessor is ready. Both value and void futures are supported.
     * If the predecessor throws, the continuation's future carries that
     * exception and fn is not invoked.
     */
    template <class T, class F>
    auto then(
        std::future<T> future,
        F&& f
    )
        -> std::future<
            typename threadpool_detail::ThenResult<
                T,
                F
            >::type
        >;


    /**
     * Submit count tasks, each invoked as f(index), and return their
     * futures in index order. Same priority and rejection semantics as
     * individual submit() calls.
     */
    template <class F>
    auto submit_batch(
        std::size_t count,
        F&& f
    )
        -> std::vector<
            std::future<
                std::invoke_result_t<F, std::size_t>
            >
        >;


    template <class F>
    auto submit_batch_priority(
        int priority,
        std::size_t count,
        F&& f
    )
        -> std::vector<
            std::future<
                std::invoke_result_t<F, std::size_t>
            >
        >;


    // ============================================================
    // Future composition (4.0.0)
    // ============================================================

    /**
     * Wait for every future in the input to complete, then return their
     * results as a vector in the same order.
     *
     * The aggregation itself runs as a task on this pool, so it does not
     * block the calling thread. That task does, however, occupy one worker
     * while it waits: if the input futures also belong to this pool, make
     * sure the pool has a spare worker for them (or use external promises),
     * otherwise the aggregation can deadlock with its own inputs.
     *
     * An empty input is fulfilled synchronously without occupying a worker.
     */
    template <class T>
    std::future<std::vector<T>> when_all(
        std::vector<std::future<T>> futures
    )
    {
        /*
         * Short-circuit the empty case. Besides avoiding a useless task,
         * this matters on a one-worker pool: a task that is only going to
         * return an empty vector must not sit behind work the caller is
         * already waiting for.
         */
        if(futures.empty())
        {
            std::promise<std::vector<T>> p;

            p.set_value(std::vector<T>{});

            return p.get_future();
        }


        /*
         * The input is held by shared_ptr rather than moved straight into
         * the lambda: std::packaged_task requires its callable to be
         * copyable, and a lambda that owns a vector of (move-only) futures
         * is not. A shared_ptr keeps the closure copyable.
         */
        auto futs =
            std::make_shared<
                std::vector<std::future<T>>
            >(std::move(futures));

        return submit(
            [futs]()
            {
                std::vector<T> out;

                out.reserve(futs->size());

                for(auto& f : *futs)
                    out.push_back(f.get());

                return out;
            }
        );
    }


    /**
     * void specialization of when_all(): waits for every future, then
     * returns once all of them have completed (or thrown).
     */
    std::future<void> when_all(
        std::vector<std::future<void>> futures
    )
    {
        if(futures.empty())
        {
            std::promise<void> p;

            p.set_value();

            return p.get_future();
        }


        auto futs =
            std::make_shared<
                std::vector<std::future<void>>
            >(std::move(futures));

        return submit(
            [futs]()
            {
                for(auto& f : *futs)
                    f.get();
            }
        );
    }


    /**
     * Wait for the first future in the input to finish (successfully or by
     * throwing), then return its index together with its result.
     *
     * Only the first finisher is reported; exceptions from the others are
     * swallowed (they are still safe to ignore because each is observed by
     * its own internally spawned task).
     *
     * Each waiter occupies one worker while it blocks. If the input
     * futures also belong to this pool, keep a spare worker for them or
     * the aggregation can deadlock with its own inputs.
     *
     * Since 5.1.1 a waiter that is discarded (clear(), stop(IMMEDIATE))
     * or rejected still spends its ticket. If every waiter disappears the
     * returned future completes with a std::runtime_error instead of
     * blocking forever.
     */
    template <class T>
    std::future<std::pair<std::size_t, T>> when_any(
        std::vector<std::future<T>> futures
    )
    {
        auto shared =
            std::make_shared<
                std::vector<std::shared_future<T>>
            >();

        shared->reserve(futures.size());

        for(auto& f : futures)
            shared->push_back(f.share());


        auto result_p =
            std::make_shared<
                std::promise<std::pair<std::size_t, T>>
            >();

        auto result_f = result_p->get_future();

        auto claimed =
            std::make_shared<std::atomic<bool>>(false);


        const std::size_t n = shared->size();

        if(n == 0)
        {
            result_p->set_exception(
                std::make_exception_ptr(
                    std::runtime_error(
                        "ThreadPool::when_any: empty input"
                    )
                )
            );

            return result_f;
        }


        /*
         * One ticket per waiter. A waiter spends its ticket either at the
         * end of the wrapper or through on_abandon when run_task() finds
         * the task already CANCELLED. Rejected submissions never enter
         * run_task() at all, so they spend theirs explicitly below.
         *
         * When the last ticket is spent without anybody claiming the
         * result, no input future can ever be observed: fulfil the promise
         * with an error rather than stranding the caller in get().
         */
        auto tickets =
            std::make_shared<std::atomic<std::size_t>>(n);

        auto finish =
            [result_p, claimed, tickets]()
            {
                if(tickets->fetch_sub(
                    1,
                    std::memory_order_acq_rel
                ) == 1)
                {
                    bool expected = false;

                    if(claimed->compare_exchange_strong(
                        expected,
                        true
                    ))
                    {
                        try
                        {
                            result_p->set_exception(
                                std::make_exception_ptr(
                                    std::runtime_error(
                                        "ThreadPool::when_any: "
                                        "all waiters were discarded"
                                    )
                                )
                            );
                        }
                        catch(...)
                        {
                        }
                    }
                }
            };


        for(std::size_t i = 0; i < n; ++i)
        {
            SubmitResult status =
                SubmitResult::ACCEPTED;


            (void)do_submit_token_ex(
                0,
                CancellationToken{},
                true,
                std::function<void()>(finish),
                true,
                status,
                [shared, result_p, claimed, finish, i]() mutable
                {
                    try
                    {
                        T v = (*shared)[i].get();

                        bool expected = false;

                        if(claimed->compare_exchange_strong(
                            expected,
                            true
                        ))
                        {
                            try
                            {
                                result_p->set_value(
                                    {i, std::move(v)}
                                );
                            }
                            catch(...)
                            {
                                try
                                {
                                    result_p->set_exception(
                                        std::current_exception()
                                    );
                                }
                                catch(...)
                                {
                                }
                            }
                        }
                    }
                    catch(...)
                    {
                        bool expected = false;

                        if(claimed->compare_exchange_strong(
                            expected,
                            true
                        ))
                        {
                            try
                            {
                                result_p->set_exception(
                                    std::current_exception()
                                );
                            }
                            catch(...)
                            {
                            }
                        }
                    }

                    finish();
                }
            );


            if(status != SubmitResult::ACCEPTED
                &&
                status != SubmitResult::CALLER_RUNS)
            {
                finish();
            }
        }


        return result_f;
    }


    /**
     * void specialization of when_any(): waits for the first future to
     * finish, then returns its index. Same discard safety as the value
     * overload (5.1.1).
     */
    std::future<std::size_t> when_any(
        std::vector<std::future<void>> futures
    )
    {
        auto shared =
            std::make_shared<
                std::vector<std::shared_future<void>>
            >();

        shared->reserve(futures.size());

        for(auto& f : futures)
            shared->push_back(f.share());


        auto result_p =
            std::make_shared<std::promise<std::size_t>>();

        auto result_f = result_p->get_future();

        auto claimed =
            std::make_shared<std::atomic<bool>>(false);


        const std::size_t n = shared->size();

        if(n == 0)
        {
            result_p->set_exception(
                std::make_exception_ptr(
                    std::runtime_error(
                        "ThreadPool::when_any: empty input"
                    )
                )
            );

            return result_f;
        }


        auto tickets =
            std::make_shared<std::atomic<std::size_t>>(n);

        auto finish =
            [result_p, claimed, tickets]()
            {
                if(tickets->fetch_sub(
                    1,
                    std::memory_order_acq_rel
                ) == 1)
                {
                    bool expected = false;

                    if(claimed->compare_exchange_strong(
                        expected,
                        true
                    ))
                    {
                        try
                        {
                            result_p->set_exception(
                                std::make_exception_ptr(
                                    std::runtime_error(
                                        "ThreadPool::when_any: "
                                        "all waiters were discarded"
                                    )
                                )
                            );
                        }
                        catch(...)
                        {
                        }
                    }
                }
            };


        for(std::size_t i = 0; i < n; ++i)
        {
            SubmitResult status =
                SubmitResult::ACCEPTED;


            (void)do_submit_token_ex(
                0,
                CancellationToken{},
                true,
                std::function<void()>(finish),
                true,
                status,
                [shared, result_p, claimed, finish, i]() mutable
                {
                    try
                    {
                        (*shared)[i].get();

                        bool expected = false;

                        if(claimed->compare_exchange_strong(
                            expected,
                            true
                        ))
                        {
                            try
                            {
                                result_p->set_value(i);
                            }
                            catch(...)
                            {
                                try
                                {
                                    result_p->set_exception(
                                        std::current_exception()
                                    );
                                }
                                catch(...)
                                {
                                }
                            }
                        }
                    }
                    catch(...)
                    {
                        bool expected = false;

                        if(claimed->compare_exchange_strong(
                            expected,
                            true
                        ))
                        {
                            try
                            {
                                result_p->set_exception(
                                    std::current_exception()
                                );
                            }
                            catch(...)
                            {
                            }
                        }
                    }

                    finish();
                }
            );


            if(status != SubmitResult::ACCEPTED
                &&
                status != SubmitResult::CALLER_RUNS)
            {
                finish();
            }
        }


        return result_f;
    }


    /**
     * Process-wide pool for quick fire-and-forget submission.
     *
     * Lazily created on first use (thread-safe magic static) and torn down
     * at program exit via a graceful stop. Prefer a dedicated pool when you
     * need to tune the thread count, idle timeout or rejection policy.
     */
    static ThreadPool& default_pool()
    {
        static ThreadPool pool(Config{});

        return pool;
    }


private:

    /*
     * TaskGroup uses the guarded submission engine directly: it needs to
     * attach a lifecycle hook and an execution-context token that the
     * public submit_token() API deliberately does not expose.
     */
    friend class TaskGroup;


    struct Task
    {
        int priority = 0;

        std::uint64_t sequence = 0;

        std::shared_ptr<std::atomic<TaskState>> state;

        /*
         * 5.0.0: the cancellation domain this task belongs to. Empty for
         * tasks submitted through the classic API.
         *
         * It lives on the queue entry (not just inside the lambda) so that
         * run_task() can install it in the thread-local execution context
         * before invoking func, on every execution path.
         */
        CancellationToken token;

        /*
         * 5.0.1: lifecycle hook for a task that will never run.
         *
         * clear(), stop(IMMEDIATE) and TaskHandle::cancel() mark a queued
         * task CANCELLED before its wrapper gets a chance to execute. A
         * wrapper that owns external bookkeeping (TaskGroup does) still
         * has to release it, otherwise wait() hangs forever. run_task()
         * invokes this hook when it detects that the wrapper did not run.
         */
        std::function<void()> on_abandon;

        std::function<void()> func;
    };


    /*
     * Result of make_task(): the ready-to-run Task plus the cancellation
     * handle / future and the owning TaskBody.
     *
     * body is handed back so the caller can abandon the packaged_task
     * (break the promise) when the task is rejected instead of enqueued.
     */
    template <class T>
    struct BuiltTask
    {
        Task task;
        TaskResult<T> result;
        std::shared_ptr<threadpool_detail::TaskBody<T>> body;
    };


    struct TaskCompare
    {
        bool operator()(
            const Task& lhs,
            const Task& rhs
        ) const
        {
            /*
             * priority_queue behavior:
             *
             * A task with a larger priority value is executed first.
             *
             * If priorities are equal:
             * the task with the smaller sequence number runs first,
             * providing FIFO ordering.
             */

            if(lhs.priority != rhs.priority)
                return lhs.priority < rhs.priority;

            return lhs.sequence > rhs.sequence;
        }
    };


    /*
     * Result of an enqueue attempt.
     */
    enum class SubmitResult
    {
        ACCEPTED,
        REJECTED_FULL,
        REJECTED_STOPPED,
        CALLER_RUNS
    };


    /*
     * Ready (runnable) task heap.
     *
     * Implemented directly on a std::vector with make_heap / push_heap /
     * pop_heap instead of std::priority_queue so that the backing storage
     * can be reserve()'d up front to max_queue. That removes the
     * reallocation (and the lock being held while every surviving element
     * is copied) that a plain priority_queue would trigger as the queue
     * grows, and caps the memory at max_queue.
     *
     * Ordering is identical to the old priority_queue: TaskCompare makes
     * the highest priority (then smallest sequence) element sit at front().
     */
    using TaskQueue =
        std::vector<Task>;


    /*
     * Worker bookkeeping.
     *
     * finished is set by the worker itself after it left worker_loop().
     * The pool uses it to reap exited std::thread objects instead of
     * letting them accumulate forever.
     *
     * local is a per-worker LIFO queue for tasks submitted from inside a
     * task that runs on this worker. Routing nested submissions there keeps
     * them off the shared global queue, so recursive / fork-join patterns
     * (parallel_for, parallel_reduce) never serialize on the global mutex.
     * The owning worker is the only producer and the only consumer, so the
     * mutex is uncontended in the common case.
     */
    /*
     * A double-ended task buffer owned by one worker.
     *
     * The owner pushes and pops from the back (LIFO), which keeps a
     * worker's own fork-join tree cache-friendly. Idle peers steal from the
     * front (FIFO) so a backlog is drained oldest-first and spread evenly.
     *
     * The mutex makes this safe for one owner + many thieves. A fully
     * lock-free Chase-Lev deque would be faster on paper, but the lock is
     * uncontended on the owner side (the common path) and only ever taken
     * briefly by a thief, so the simpler design gives the same scaling
     * without the ABA hazards.
     */
    struct WorkStealingDeque
    {
        void push(Task t)
        {
            std::lock_guard<std::mutex> lock(mtx_);

            q_.push_back(std::move(t));
        }


        /*
         * Owner side: take the most recently pushed task.
         */
        bool pop(Task& out)
        {
            std::lock_guard<std::mutex> lock(mtx_);

            if(q_.empty())
                return false;

            out = std::move(q_.back());
            q_.pop_back();

            return true;
        }


        /*
         * Thief side: take the oldest task.
         */
        bool steal(Task& out)
        {
            std::lock_guard<std::mutex> lock(mtx_);

            if(q_.empty())
                return false;

            out = std::move(q_.front());
            q_.pop_front();

            return true;
        }


        bool empty() const
        {
            std::lock_guard<std::mutex> lock(mtx_);

            return q_.empty();
        }


        std::size_t size() const
        {
            std::lock_guard<std::mutex> lock(mtx_);

            return q_.size();
        }


        mutable std::mutex mtx_;

        std::deque<Task> q_;
    };


    struct Worker
    {
        std::thread thread;

        std::atomic<bool> finished{false};

        /*
         * The worker's own task buffer. Nested submissions (submit() called
         * from inside a task) land here; idle peers reach in to steal.
         */
        WorkStealingDeque deque_;
    };


private:

    /**
     * Common implementation shared by the throwing and non-throwing
     * submission APIs.
     *
     * allow_wait:
     *   permits blocking on a full queue when the policy is BLOCK.
     *
     * A status other than ACCEPTED / CALLER_RUNS hands the reject callback
     * back through reject_cb so it can run outside of the lock.
     */
    SubmitResult enqueue_task(
        Task& task,
        bool allow_wait,
        std::function<void()>& reject_cb
    );


    template <class F, class... Args>
    auto do_submit(
        int priority,
        bool allow_wait,
        SubmitResult& status,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    /*
     * The real submission engine since 5.0.0. do_submit() forwards here
     * with an empty token, so the classic API keeps its exact behaviour
     * while submit_token() can attach a cancellation domain.
     */
    template <class F, class... Args>
    auto do_submit_token(
        int priority,
        CancellationToken token,
        bool allow_wait,
        SubmitResult& status,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    /*
     * The full engine behind do_submit_token().
     *
     * skip_on_cancel:
     *   true  - a cancelled token makes the packaged_task throw
     *           TaskCancelled before user code (submit_token semantics);
     *   false - the token is only installed in the execution context, the
     *           callable always runs and decides what cancellation means.
     *
     * on_abandon:
     *   invoked by run_task() when the wrapper never ran (clear(),
     *   stop(IMMEDIATE), TaskHandle::cancel(), submission rejected).
     */
    template <class F, class... Args>
    auto do_submit_token_ex(
        int priority,
        CancellationToken token,
        bool skip_on_cancel,
        std::function<void()> on_abandon,
        bool allow_wait,
        SubmitResult& status,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    /*
     * Builds the Task, its future and its cancellation handle without
     * touching any queue.
     *
     * Shared by do_submit() (immediate) and do_schedule_at() (deferred) so
     * the callable binding / move-only handling lives in exactly one place.
     */
    template <class F, class... Args>
    auto make_task(
        int priority,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    template <class F, class... Args>
    auto do_schedule_at(
        int priority,
        bool allow_wait,
        SubmitResult& status,
        std::chrono::steady_clock::time_point at,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    /*
     * Token-aware counterpart of do_schedule_at(); the classic path
     * forwards to it with an empty token.
     */
    template <class F, class... Args>
    auto do_schedule_at_token(
        int priority,
        CancellationToken token,
        bool allow_wait,
        SubmitResult& status,
        std::chrono::steady_clock::time_point at,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    template <class Rep, class Period, class F, class... Args>
    auto do_schedule_after(
        int priority,
        bool allow_wait,
        SubmitResult& status,
        const std::chrono::duration<
            Rep,
            Period
        >& delay,
        F&& f,
        Args&&... args
    )
        -> TaskResult<
            std::invoke_result_t<F, Args...>
        >;


    bool pop_task_locked(Task& out);

    /*
     * Moves every queued task into dropped and returns how many there
     * were.
     *
     * The tasks must be handed to run_task() after releasing the lock:
     * that is where the promise becomes broken_promise and where the
     * cancellation is counted.
     */
    std::size_t discard_tasks_locked(
        std::vector<Task>& dropped
    );

    void run_task(Task& task);

    void run_dropped(std::vector<Task>& dropped);

    void mark_exit_locked();

    void worker_loop(Worker* self);

    /*
     * Snapshot the worker list under mtx_, then try to steal one task from
     * a peer's deque (skipping self). Returns true and fills out on success.
     *
     * Called WITHOUT mtx_ held: the snapshot keeps every candidate Worker
     * alive (workers_ now holds shared_ptr) so a victim that is reaped and
     * joined while we are stealing from it cannot be a dangling pointer.
     */
    bool try_steal(
        Task& out,
        Worker* self
    );

    /*
     * Account for n tasks finishing: drop them from inflight_ and, if that
     * brings inflight_ to zero and every queue is empty, release the
     * threads parked in wait().
     *
     * The increment side (in worker_loop) adds to inflight_ BEFORE removing
     * the task from its deque, so a waiter never sees a task that is in
     * neither counter.
     */
    void complete_tasks(
        std::size_t n
    );

    /*
     * Creates the Worker object (and updates the counters) while holding
     * mtx_, but does NOT start the OS thread. The caller spawns it outside
     * the lock via spawn_worker() so that thread creation never blocks
     * every other submitter.
     *
     * Returns the new Worker, or nullptr when stopped / at max_threads.
     */
    Worker* create_worker_locked();

    /*
     * Decides whether to grow, creating a Worker object under mtx_ when it
     * does. Reaps exited workers into out (without joining). Returns the
     * newly created Worker (to be spawned outside the lock), or nullptr.
     */
    Worker* maybe_expand_locked(
        std::vector<std::shared_ptr<Worker>>& out
    );

    /*
     * Collects exited workers into out (erasing them from workers_) without
     * joining their threads. The caller joins them outside of mtx_ so that a
     * join() - which can block for milliseconds - never stalls the pool.
     */
    void reap_workers_locked(
        std::vector<std::shared_ptr<Worker>>& out
    );

    /*
     * Starts the OS thread for a Worker created by create_worker_locked().
     *
     * Called WITHOUT mtx_ held.
     */
    void spawn_worker(Worker* worker);

    /*
     * Undoes create_worker_locked() when spawn_worker() failed: removes the
     * Worker from workers_ and restores the counters. Caller holds mtx_.
     */
    void rollback_worker_locked(Worker* worker);


    /*
     * Move every scheduled task whose deadline has passed into tasks_ (the
     * runnable priority queue), preserving its priority.
     *
     * The caller must hold mtx_. Never promotes while paused_: a paused pool
     * must not start draining deferred work.
     */
    void promote_scheduled_locked();


    /*
     * Smallest scheduled deadline, or steady_clock::time_point::max() when
     * nothing is scheduled. Caller must hold mtx_.
     */
    std::chrono::steady_clock::time_point
        next_scheduled_locked() const;


    bool has_due_scheduled_locked() const;


    /*
     * wait() and friends count the calling task as in-flight work, so a
     * call from a worker of the same pool can never be satisfied.
     *
     * Such a call is a programming error. Detecting it turns an
     * unconditional hang into an immediate std::logic_error.
     */
    void ensure_not_worker(
        const char* what
    ) const;


private:

    Config cfg_;


    TaskQueue tasks_;


    /*
     * Deferred tasks, keyed by their steady_clock deadline.
     *
     * Guarded by mtx_. promote_scheduled_locked() moves the due ones into
     * tasks_ when a worker wakes up; clear() / stop() drain it as well.
     */
    std::multimap<
        std::chrono::steady_clock::time_point,
        Task
    > scheduled_;


    /*
     * Each entry is heap allocated so that Worker is never reallocated:
     * a running worker hands its own address to the thread entry point.
     *
     * Held by shared_ptr (not unique_ptr) so a thief that snapshots the
     * list and then steals from a peer cannot end up with a dangling
     * pointer if that peer is reaped and joined in the meantime. The
     * refcount keeps the Worker - and its deque_ mutex - alive until the
     * thief is done.
     *
     * Entries are removed by reap_workers_locked() once a worker exited.
     */
    std::vector<
        std::shared_ptr<Worker>
    > workers_;


    mutable std::mutex mtx_;

    std::condition_variable cv_task_;

    std::condition_variable cv_queue_;

    std::condition_variable cv_idle_;


    bool stopping_ = false;

    bool paused_ = false;

    PoolStopMode stop_mode_ =
        PoolStopMode::GRACEFUL;


    std::size_t worker_count_ = 0;

    std::size_t idle_count_ = 0;


    /*
     * Number of tasks already taken out of the queue but not finished.
     *
     * Without it, wait() could return right after a worker popped the last
     * task but before that task actually ran.
     */
    std::atomic<std::size_t> inflight_{0};


    /*
     * Tasks sitting in a worker's local deque but not yet executed.
     *
     * Incremented when a task is pushed to a local deque (always by the
     * owning worker) and decremented when any worker pops it. Together with
     * inflight_ and the two queues it feeds the wait() predicate.
     */
    std::atomic<std::size_t> local_pending_{0};


    /*
     * Number of workers that left worker_loop() but whose thread has not
     * been joined yet.
     *
     * Used as a cheap "is there anything to reap" test so that a submit
     * does not have to walk workers_ every time.
     */
    std::size_t exited_pending_ = 0;


    /*
     * Largest queue length seen so far.
     */
    std::size_t peak_queue_ = 0;


    /*
     * How many worker threads this pool started in total.
     *
     * Together with worker_count it shows how often the pool had to grow
     * and shrink.
     */
    std::uint64_t workers_created_ = 0;


    std::function<void()> reject_callback_;


    std::atomic<std::uint64_t> sequence_{0};

    std::atomic<std::uint64_t> submitted_{0};

    std::atomic<std::uint64_t> completed_{0};

    std::atomic<std::uint64_t> cancelled_{0};

    std::atomic<std::uint64_t> rejected_{0};

    std::atomic<std::uint64_t> running_{0};

    std::atomic<std::uint64_t> stolen_{0};


    /*
     * 5.0.0 counters. failed_ is incremented by the task wrapper itself,
     * before std::packaged_task stores the exception in the future, so a
     * caller that never calls get() still leaves a trace in the metrics.
     */
    std::atomic<std::uint64_t> failed_{0};

    std::atomic<std::uint64_t> token_cancelled_{0};

    std::atomic<std::uint64_t> timed_out_{0};
};


// ============================================================
// submit
// ============================================================

template <class F, class... Args>
auto ThreadPool::submit(
    F&& f,
    Args&&... args
)
    -> std::future<
        std::invoke_result_t<F, Args...>
    >
{
    return submit_priority(
        0,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// submit_priority
// ============================================================

template <class F, class... Args>
auto ThreadPool::submit_priority(
    int priority,
    F&& f,
    Args&&... args
)
    -> std::future<
        std::invoke_result_t<F, Args...>
    >
{
    auto result = submit_cancellable(
        priority,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );

    return std::move(result.future);
}


// ============================================================
// submit_cancellable
// ============================================================

template <class F, class... Args>
auto ThreadPool::submit_cancellable(
    int priority,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    SubmitResult status =
        SubmitResult::ACCEPTED;


    auto result = do_submit(
        priority,
        true,
        status,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(status == SubmitResult::REJECTED_STOPPED)
    {
        throw std::runtime_error(
            "ThreadPool already stopped"
        );
    }


    if(status == SubmitResult::REJECTED_FULL)
    {
        throw std::runtime_error(
            "ThreadPool queue full, task rejected"
        );
    }


    return result;
}


// ============================================================
// try_submit
// ============================================================

template <class F, class... Args>
auto ThreadPool::try_submit(
    F&& f,
    Args&&... args
)
    -> std::optional<
        std::future<
            std::invoke_result_t<F, Args...>
        >
    >
{
    return try_submit_priority(
        0,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// try_submit_priority
// ============================================================

template <class F, class... Args>
auto ThreadPool::try_submit_priority(
    int priority,
    F&& f,
    Args&&... args
)
    -> std::optional<
        std::future<
            std::invoke_result_t<F, Args...>
        >
    >
{
    auto result = try_submit_cancellable(
        priority,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(!result)
        return std::nullopt;


    return std::move(result->future);
}


// ============================================================
// try_submit_cancellable
// ============================================================

template <class F, class... Args>
auto ThreadPool::try_submit_cancellable(
    int priority,
    F&& f,
    Args&&... args
)
    -> std::optional<
        TaskResult<
            std::invoke_result_t<F, Args...>
        >
    >
{
    SubmitResult status =
        SubmitResult::ACCEPTED;


    auto result = do_submit(
        priority,
        false,
        status,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(status != SubmitResult::ACCEPTED
        &&
        status != SubmitResult::CALLER_RUNS)
    {
        return std::nullopt;
    }


    return result;
}


// ============================================================
// wait_for
// ============================================================

template <class Rep, class Period>
bool ThreadPool::wait_for(
    const std::chrono::duration<
        Rep,
        Period
    >& timeout
)
{
    ensure_not_worker("wait_for");


    std::unique_lock<std::mutex> lock(mtx_);


    return cv_idle_.wait_for(
        lock,
        timeout,
        [this]()
        {
            return tasks_.empty()
                &&
                inflight_.load(
                    std::memory_order_relaxed
                ) == 0
                &&
                scheduled_.empty()
                &&
                local_pending_.load(
                    std::memory_order_relaxed
                ) == 0;
        }
    );
}


// ============================================================
// wait_until
// ============================================================

template <class Clock, class Duration>
bool ThreadPool::wait_until(
    const std::chrono::time_point<
        Clock,
        Duration
    >& deadline
)
{
    ensure_not_worker("wait_until");


    std::unique_lock<std::mutex> lock(mtx_);


    return cv_idle_.wait_until(
        lock,
        deadline,
        [this]()
        {
            return tasks_.empty()
                &&
                inflight_.load(
                    std::memory_order_relaxed
                ) == 0
                &&
                scheduled_.empty()
                &&
                local_pending_.load(
                    std::memory_order_relaxed
                ) == 0;
        }
    );
}


// ============================================================
// do_submit
// ============================================================

template <class F, class... Args>
auto ThreadPool::do_submit_token_ex(
    int priority,
    CancellationToken token,
    bool skip_on_cancel,
    std::function<void()> on_abandon,
    bool allow_wait,
    SubmitResult& status,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    using Callable =
        std::decay_t<F>;

    using ArgsTuple =
        std::tuple<std::decay_t<Args>...>;


    static_assert(
        std::is_invocable_v<
            Callable&,
            std::decay_t<Args>...
        >,
        "ThreadPool: callable is not invocable with the given arguments"
    );


    using ReturnType =
        std::invoke_result_t<F, Args...>;

    using Body =
        threadpool_detail::TaskBody<ReturnType>;


    /*
     * Bind the callable and its arguments into a zero-argument function.
     *
     * A lambda + tuple is used instead of std::bind because std::bind
     * always decays and copies its arguments. That would reject move-only
     * arguments and introduce unnecessary copies.
     *
     * std::apply invokes through std::invoke, so member function pointers
     * are supported as well.
     *
     * 5.0.0: the cancellation token is captured here (inside the
     * packaged_task callable) as well as on the queue entry. Throwing
     * before the user callable is entered lets packaged_task deliver
     * TaskCancelled through the future; the outer wrapper only has to
     * update the state and the counters.
     *
     * skip_on_cancel is false for TaskGroup: there the wrapper itself
     * owns the cancellation check, so skipping it would strand the
     * group's pending counter.
     */
    auto body =
        std::make_shared<Body>(
            [
                this,
                token,
                skip_on_cancel,
                callable =
                    Callable(
                        std::forward<F>(f)
                    ),
                bound =
                    ArgsTuple(
                        std::forward<Args>(args)...
                    )
            ]() mutable -> ReturnType
            {
                if(skip_on_cancel)
                    token.throw_if_cancelled();


                try
                {
                    if constexpr(
                        std::is_void_v<ReturnType>
                    )
                    {
                        std::apply(
                            std::move(callable),
                            std::move(bound)
                        );
                    }
                    else
                    {
                        return std::apply(
                            std::move(callable),
                            std::move(bound)
                        );
                    }
                }
                catch(...)
                {
                    /*
                     * packaged_task will store this exception in the future.
                     * Count it here too: callers that never call get() still
                     * leave a trace in PoolMetrics.failed.
                     */
                    failed_.fetch_add(
                        1,
                        std::memory_order_relaxed
                    );

                    throw;
                }
            }
        );


    auto future = body->task.get_future();


    /*
     * Aliasing shared_ptr: points at TaskBody::state but keeps the whole
     * body alive.
     *
     * TaskHandle, the queue entry and the worker lambda all share this one
     * control block.
     */
    std::shared_ptr<
        std::atomic<TaskState>
    > state(
        body,
        &body->state
    );


    TaskHandle handle(state);


    Task task;

    task.priority = priority;

    task.sequence =
        sequence_.fetch_add(
            1,
            std::memory_order_relaxed
        );

    task.state = state;


    /*
     * Capturing the single body pointer keeps this lambda small, so it
     * usually fits into the std::function small buffer instead of
     * allocating.
     */
    task.token = token;

    task.on_abandon = std::move(on_abandon);

    task.func =
        [this, body, token]()
        {
            std::atomic<TaskState>& state_ref =
                body->state;


            /*
             * Transition from WAITING to RUNNING.
             *
             * If cancel() succeeds first, the state will already be
             * CANCELLED and the worker will skip task execution.
             */
            TaskState expected =
                TaskState::WAITING;


            if(!state_ref.compare_exchange_strong(
                expected,
                TaskState::RUNNING,
                std::memory_order_acq_rel
            ))
            {
                if(expected == TaskState::CANCELLED)
                {
                    cancelled_.fetch_add(
                        1,
                        std::memory_order_relaxed
                    );
                }


                /*
                 * This promise will never be fulfilled.
                 *
                 * The handle shares ownership of TaskBody, so dropping the
                 * queue entry alone would not destroy the packaged_task.
                 * Resetting it here is what turns the caller's future into
                 * broken_promise instead of blocking it forever.
                 */
                body->task.reset();

                return;
            }


            /*
             * Read the token when the task actually starts. A cancellation
             * that arrives while the task is queued is what we act on here;
             * the packaged_task callable re-checks it a moment later and
             * throws TaskCancelled before entering user code.
             */
            const bool cancelled =
                token.is_cancelled();


            running_.fetch_add(
                1,
                std::memory_order_relaxed
            );


            /*
             * std::packaged_task captures exceptions thrown by the user task
             * and stores them in the associated future.
             *
             * Therefore, exceptions from the user callable normally do not
             * escape from operator() here.
             */
            try
            {
                body->task();
            }
            catch(...)
            {
                /*
                 * Prevent the worker thread from terminating in case of
                 * an unexpected exception.
                 */
            }


            running_.fetch_sub(
                1,
                std::memory_order_relaxed
            );


            if(cancelled)
            {
                /*
                 * A deadline-driven cancellation is still a cancellation,
                 * but callers tuning timeout budgets need to see it
                 * separately. This counts tasks that were already expired
                 * when they started; a deadline reached while a task is
                 * running is only observable to that task itself.
                 */
                if(token.timed_out())
                {
                    timed_out_.fetch_add(
                        1,
                        std::memory_order_relaxed
                    );
                }


                token_cancelled_.fetch_add(
                    1,
                    std::memory_order_relaxed
                );

                cancelled_.fetch_add(
                    1,
                    std::memory_order_relaxed
                );

                state_ref.store(
                    TaskState::CANCELLED,
                    std::memory_order_release
                );
            }
            else
            {
                state_ref.store(
                    TaskState::COMPLETED,
                    std::memory_order_release
                );

                completed_.fetch_add(
                    1,
                    std::memory_order_relaxed
                );
            }
        };


    std::function<void()> reject_callback;


    SubmitResult result =
        enqueue_task(
            task,
            allow_wait,
            reject_callback
        );


    if(result == SubmitResult::ACCEPTED)
    {
        /*
         * Nothing to do here: the worker notification - if any - already
         * happened inside enqueue_task(), while the lock was still held.
         */
    }
    else if(result == SubmitResult::CALLER_RUNS)
    {
        /*
         * The queue is full and the policy asks for caller runs.
         *
         * Execute the task here, outside of any lock, so that a blocked
         * producer still makes progress.
         */
        run_task(task);
    }
    else
    {
        /*
         * Rejected: nothing will ever run this task, so the future must
         * not be left waiting for a value that never arrives.
         *
         * The handle keeps TaskBody alive, so the packaged_task has to be
         * abandoned explicitly to break the promise.
         */
        body->task.reset();


        /*
         * Inform the caller through the reject callback.
         * It runs outside of the internal lock.
         */
        if(reject_callback)
        {
            try
            {
                reject_callback();
            }
            catch(...)
            {
                /*
                 * An exception from the rejection callback must not
                 * affect the thread pool.
                 */
            }
        }
    }


    status = result;


    return TaskResult<ReturnType>{
        std::move(future),
        std::move(handle)
    };
}


// ============================================================
// do_submit_token (public API defaults: skip on cancel, no hook)
// ============================================================

template <class F, class... Args>
auto ThreadPool::do_submit_token(
    int priority,
    CancellationToken token,
    bool allow_wait,
    SubmitResult& status,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    return do_submit_token_ex(
        priority,
        std::move(token),
        true,
        std::function<void()>{},
        allow_wait,
        status,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// do_submit (classic API, forwards to the token-aware engine)
// ============================================================

template <class F, class... Args>
auto ThreadPool::do_submit(
    int priority,
    bool allow_wait,
    SubmitResult& status,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    return do_submit_token(
        priority,
        CancellationToken{},
        allow_wait,
        status,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// do_schedule_at
// ============================================================

template <class F, class... Args>
auto ThreadPool::do_schedule_at_token(
    int priority,
    CancellationToken token,
    bool allow_wait,
    SubmitResult& status,
    std::chrono::steady_clock::time_point at,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    using Callable =
        std::decay_t<F>;

    using ArgsTuple =
        std::tuple<std::decay_t<Args>...>;

    using ReturnType =
        std::invoke_result_t<F, Args...>;

    using Body =
        threadpool_detail::TaskBody<ReturnType>;


    static_assert(
        std::is_invocable_v<
            Callable&,
            std::decay_t<Args>...
        >,
        "ThreadPool: callable is not invocable with the given arguments"
    );


    /*
     * Deferred tasks are not subject to max_queue, so the BLOCK policy's
     * allow_wait flag has nothing to do here. It is kept in the signature
     * to mirror do_submit_token() and the public scheduling API.
     */
    (void)allow_wait;


    /*
     * Same cooperative-cancellation contract as the immediate path: a
     * token cancelled before the deadline expires makes the future carry
     * TaskCancelled instead of a value.
     */
    auto body =
        std::make_shared<Body>(
            [
                this,
                token,
                callable =
                    Callable(
                        std::forward<F>(f)
                    ),
                bound =
                    ArgsTuple(
                        std::forward<Args>(args)...
                    )
            ]() mutable -> ReturnType
            {
                token.throw_if_cancelled();


                try
                {
                    if constexpr(
                        std::is_void_v<ReturnType>
                    )
                    {
                        std::apply(
                            std::move(callable),
                            std::move(bound)
                        );
                    }
                    else
                    {
                        return std::apply(
                            std::move(callable),
                            std::move(bound)
                        );
                    }
                }
                catch(...)
                {
                    failed_.fetch_add(
                        1,
                        std::memory_order_relaxed
                    );

                    throw;
                }
            }
        );


    auto future = body->task.get_future();


    std::shared_ptr<
        std::atomic<TaskState>
    > state(
        body,
        &body->state
    );


    TaskHandle handle(state);


    Task task;

    task.priority = priority;

    task.sequence =
        sequence_.fetch_add(
            1,
            std::memory_order_relaxed
        );

    task.state = state;


    task.token = token;

    task.func =
        [this, body, token]()
        {
            std::atomic<TaskState>& state_ref =
                body->state;


            TaskState expected =
                TaskState::WAITING;


            if(!state_ref.compare_exchange_strong(
                expected,
                TaskState::RUNNING,
                std::memory_order_acq_rel
            ))
            {
                if(expected == TaskState::CANCELLED)
                {
                    cancelled_.fetch_add(
                        1,
                        std::memory_order_relaxed
                    );
                }


                body->task.reset();

                return;
            }


            const bool cancelled =
                token.is_cancelled();


            running_.fetch_add(
                1,
                std::memory_order_relaxed
            );


            try
            {
                body->task();
            }
            catch(...)
            {
            }


            running_.fetch_sub(
                1,
                std::memory_order_relaxed
            );


            if(cancelled)
            {
                /*
                 * A deadline-driven cancellation is still a cancellation,
                 * but callers tuning timeout budgets need to see it
                 * separately. This counts tasks that were already expired
                 * when they started; a deadline reached while a task is
                 * running is only observable to that task itself.
                 */
                if(token.timed_out())
                {
                    timed_out_.fetch_add(
                        1,
                        std::memory_order_relaxed
                    );
                }


                token_cancelled_.fetch_add(
                    1,
                    std::memory_order_relaxed
                );

                cancelled_.fetch_add(
                    1,
                    std::memory_order_relaxed
                );

                state_ref.store(
                    TaskState::CANCELLED,
                    std::memory_order_release
                );
            }
            else
            {
                state_ref.store(
                    TaskState::COMPLETED,
                    std::memory_order_release
                );

                completed_.fetch_add(
                    1,
                    std::memory_order_relaxed
                );
            }
        };


    std::function<void()> reject_callback;


    {
        std::lock_guard<std::mutex> lock(mtx_);


        /*
         * A stopped pool cannot accept deferred work either: failing to do
         * so would let a future hang forever waiting for a task that the
         * pool will never run.
         */
        if(stopping_)
        {
            rejected_.fetch_add(
                1,
                std::memory_order_relaxed
            );


            reject_callback = reject_callback_;


            status = SubmitResult::REJECTED_STOPPED;


            body->task.reset();


            if(reject_callback)
            {
                try
                {
                    reject_callback();
                }
                catch(...)
                {
                }
            }


            return TaskResult<ReturnType>{
                std::move(future),
                std::move(handle)
            };
        }


        /*
         * Deferred tasks are not subject to max_queue: they are time-bound,
         * not pressure-bound, and counting them against the pressure limit
         * would make submit_after() surprising.
         */
        scheduled_.emplace(
            at,
            std::move(task)
        );

        submitted_.fetch_add(
            1,
            std::memory_order_relaxed
        );
    }


    /*
     * Wake idle workers so they recompute their wait deadline. A newly
     * scheduled task with an earlier deadline than the one a worker is
     * already waiting on must be noticed.
     */
    cv_task_.notify_all();


    status = SubmitResult::ACCEPTED;


    return TaskResult<ReturnType>{
        std::move(future),
        std::move(handle)
    };
}


// ============================================================
// do_schedule_at (classic API, forwards to the token-aware engine)
// ============================================================

template <class F, class... Args>
auto ThreadPool::do_schedule_at(
    int priority,
    bool allow_wait,
    SubmitResult& status,
    std::chrono::steady_clock::time_point at,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    return do_schedule_at_token(
        priority,
        CancellationToken{},
        allow_wait,
        status,
        at,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// do_schedule_after
// ============================================================

template <class Rep, class Period, class F, class... Args>
auto ThreadPool::do_schedule_after(
    int priority,
    bool allow_wait,
    SubmitResult& status,
    const std::chrono::duration<
        Rep,
        Period
    >& delay,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    auto at =
        std::chrono::steady_clock::now()
        + delay;


    return do_schedule_at(
        priority,
        allow_wait,
        status,
        at,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// submit_at / submit_after
// ============================================================

template <class F, class... Args>
auto ThreadPool::submit_at(
    std::chrono::steady_clock::time_point at,
    F&& f,
    Args&&... args
)
    -> std::future<
        std::invoke_result_t<F, Args...>
    >
{
    SubmitResult status =
        SubmitResult::ACCEPTED;


    auto result = do_schedule_at(
        0,
        true,
        status,
        at,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(status == SubmitResult::REJECTED_STOPPED)
    {
        throw std::runtime_error(
            "ThreadPool already stopped"
        );
    }


    return std::move(result.future);
}


template <class Rep, class Period, class F, class... Args>
auto ThreadPool::submit_after(
    const std::chrono::duration<
        Rep,
        Period
    >& delay,
    F&& f,
    Args&&... args
)
    -> std::future<
        std::invoke_result_t<F, Args...>
    >
{
    return submit_at(
        std::chrono::steady_clock::now()
        + delay,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


template <class F, class... Args>
auto ThreadPool::submit_cancellable_at(
    std::chrono::steady_clock::time_point at,
    int priority,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    SubmitResult status =
        SubmitResult::ACCEPTED;


    auto result = do_schedule_at(
        priority,
        true,
        status,
        at,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(status == SubmitResult::REJECTED_STOPPED)
    {
        throw std::runtime_error(
            "ThreadPool already stopped"
        );
    }


    return result;
}


template <class Rep, class Period, class F, class... Args>
auto ThreadPool::submit_cancellable_after(
    const std::chrono::duration<
        Rep,
        Period
    >& delay,
    int priority,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    return submit_cancellable_at(
        std::chrono::steady_clock::now()
        + delay,
        priority,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


template <class F, class... Args>
auto ThreadPool::try_submit_at(
    std::chrono::steady_clock::time_point at,
    int priority,
    F&& f,
    Args&&... args
)
    -> std::optional<
        std::future<
            std::invoke_result_t<F, Args...>
        >
    >
{
    SubmitResult status =
        SubmitResult::ACCEPTED;


    auto result = do_schedule_at(
        priority,
        false,
        status,
        at,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(status != SubmitResult::ACCEPTED)
        return std::nullopt;


    return std::move(result.future);
}


template <class Rep, class Period, class F, class... Args>
auto ThreadPool::try_submit_after(
    const std::chrono::duration<
        Rep,
        Period
    >& delay,
    int priority,
    F&& f,
    Args&&... args
)
    -> std::optional<
        std::future<
            std::invoke_result_t<F, Args...>
        >
    >
{
    return try_submit_at(
        std::chrono::steady_clock::now()
        + delay,
        priority,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


template <class F, class... Args>
auto ThreadPool::try_submit_cancellable_at(
    std::chrono::steady_clock::time_point at,
    int priority,
    F&& f,
    Args&&... args
)
    -> std::optional<
        TaskResult<
            std::invoke_result_t<F, Args...>
        >
    >
{
    SubmitResult status =
        SubmitResult::ACCEPTED;


    auto result = do_schedule_at(
        priority,
        false,
        status,
        at,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(status != SubmitResult::ACCEPTED)
        return std::nullopt;


    return result;
}


template <class Rep, class Period, class F, class... Args>
auto ThreadPool::try_submit_cancellable_after(
    const std::chrono::duration<
        Rep,
        Period
    >& delay,
    int priority,
    F&& f,
    Args&&... args
)
    -> std::optional<
        TaskResult<
            std::invoke_result_t<F, Args...>
        >
    >
{
    return try_submit_cancellable_at(
        std::chrono::steady_clock::now()
        + delay,
        priority,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// parallel_for
// ============================================================

namespace threadpool_detail
{
    template <class T>
    struct ParallelState
    {
        std::mutex mtx;

        std::exception_ptr ex;

        bool has_ex = false;

        std::atomic<std::size_t> remaining{0};

        std::promise<T> promise;

        T acc;
    };


    template <>
    struct ParallelState<void>
    {
        std::mutex mtx;

        std::exception_ptr ex;

        bool has_ex = false;

        std::atomic<std::size_t> remaining{0};

        std::promise<void> promise;
    };
}


template <class F>
std::future<void> ThreadPool::parallel_for(
    std::size_t begin,
    std::size_t end,
    std::size_t grainsize,
    F&& f
)
{
    if(grainsize == 0)
        grainsize = 1;


    ensure_not_worker("parallel_for");


    const std::size_t span =
        end > begin ? end - begin : 0;

    const std::size_t chunks =
        span == 0 ? 0 : (span + grainsize - 1) / grainsize;


    auto state =
        std::make_shared<
            threadpool_detail::ParallelState<void>
        >();


    if(chunks == 0)
    {
        state->promise.set_value();

        return state->promise.get_future();
    }


    state->remaining.store(chunks);


    for(std::size_t i = 0; i < chunks; ++i)
    {
        const std::size_t chunk_begin =
            begin + i * grainsize;

        const std::size_t chunk_end =
            chunk_begin + grainsize < end
                ? chunk_begin + grainsize
                : end;


        this->submit(
            [this, state, chunk_begin, chunk_end, &f]()
            {
                try
                {
                    for(std::size_t idx = chunk_begin;
                        idx < chunk_end;
                        ++idx)
                    {
                        f(idx);
                    }
                }
                catch(...)
                {
                    std::lock_guard<std::mutex> lock(
                        state->mtx
                    );

                    if(!state->has_ex)
                    {
                        state->ex =
                            std::current_exception();

                        state->has_ex = true;
                    }
                }


                if(state->remaining.fetch_sub(1) == 1)
                {
                    std::lock_guard<std::mutex> lock(
                        state->mtx
                    );

                    if(state->has_ex)
                        state->promise.set_exception(
                            state->ex
                        );
                    else
                        state->promise.set_value();
                }
            }
        );
    }


    return state->promise.get_future();
}


template <class F>
std::future<void> ThreadPool::parallel_for(
    std::size_t begin,
    std::size_t end,
    F&& f
)
{
    const std::size_t n =
        end > begin ? end - begin : 0;


    /*
     * Aim for a few chunks per worker so the load is spread but the
     * per-chunk overhead stays small.
     */
    const std::size_t grainsize =
        n == 0 ? 1 : (n + 31) / 32;


    return parallel_for(
        begin,
        end,
        grainsize,
        std::forward<F>(f)
    );
}


// ============================================================
// parallel_reduce
// ============================================================

template <class T, class F, class BinOp>
std::future<T> ThreadPool::parallel_reduce(
    std::size_t begin,
    std::size_t end,
    std::size_t grainsize,
    T init,
    F&& f,
    BinOp&& op
)
{
    if(grainsize == 0)
        grainsize = 1;


    ensure_not_worker("parallel_reduce");


    const std::size_t span =
        end > begin ? end - begin : 0;

    const std::size_t chunks =
        span == 0 ? 0 : (span + grainsize - 1) / grainsize;


    auto state =
        std::make_shared<
            threadpool_detail::ParallelState<T>
        >();


    if(chunks == 0)
    {
        state->promise.set_value(
            std::move(init)
        );

        return state->promise.get_future();
    }


    state->remaining.store(chunks);

    state->acc = std::move(init);


    for(std::size_t i = 0; i < chunks; ++i)
    {
        const std::size_t chunk_begin =
            begin + i * grainsize;

        const std::size_t chunk_end =
            chunk_begin + grainsize < end
                ? chunk_begin + grainsize
                : end;


        this->submit(
            [this, state, chunk_begin, chunk_end, &f, &op]()
            {
                try
                {
                    T partial =
                        f(
                            chunk_begin,
                            chunk_end
                        );


                    std::lock_guard<std::mutex> lock(
                        state->mtx
                    );


                    if(!state->has_ex)
                    {
                        state->acc =
                            op(
                                std::move(state->acc),
                                std::move(partial)
                            );
                    }
                }
                catch(...)
                {
                    std::lock_guard<std::mutex> lock(
                        state->mtx
                    );


                    if(!state->has_ex)
                    {
                        state->ex =
                            std::current_exception();

                        state->has_ex = true;
                    }
                }


                if(state->remaining.fetch_sub(1) == 1)
                {
                    std::lock_guard<std::mutex> lock(
                        state->mtx
                    );

                    if(state->has_ex)
                        state->promise.set_exception(
                            state->ex
                        );
                    else
                        state->promise.set_value(
                            std::move(state->acc)
                        );
                }
            }
        );
    }


    return state->promise.get_future();
}


// ============================================================
// submit_cancellable_token (5.0.0)
// ============================================================

template <class F, class... Args>
auto ThreadPool::submit_cancellable_token(
    int priority,
    CancellationToken token,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    SubmitResult status =
        SubmitResult::ACCEPTED;


    auto result = do_submit_token(
        priority,
        std::move(token),
        true,
        status,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(status == SubmitResult::REJECTED_STOPPED)
    {
        throw std::runtime_error(
            "ThreadPool already stopped"
        );
    }


    if(status == SubmitResult::REJECTED_FULL)
    {
        throw std::runtime_error(
            "ThreadPool queue full, task rejected"
        );
    }


    return result;
}


// ============================================================
// submit_token_priority / submit_token
// ============================================================

template <class F, class... Args>
auto ThreadPool::submit_token_priority(
    int priority,
    CancellationToken token,
    F&& f,
    Args&&... args
)
    -> std::future<
        std::invoke_result_t<F, Args...>
    >
{
    auto result = submit_cancellable_token(
        priority,
        std::move(token),
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    return std::move(result.future);
}


template <class F, class... Args>
auto ThreadPool::submit_token(
    CancellationToken token,
    F&& f,
    Args&&... args
)
    -> std::future<
        std::invoke_result_t<F, Args...>
    >
{
    return submit_token_priority(
        0,
        std::move(token),
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// submit_with_deadline_priority / submit_with_timeout_priority (5.1.0)
// ============================================================

template <class F, class... Args>
auto ThreadPool::submit_with_deadline_priority(
    int priority,
    std::chrono::steady_clock::time_point deadline,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    /*
     * The source is a local, but the token holds the shared state. Every
     * queued task keeps that state alive until its wrapper has run, so
     * the deadline outlives this function without any registry.
     */
    CancellationSource source;

    source.cancel_at(deadline);


    return submit_cancellable_token(
        priority,
        source.token(),
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


template <class Rep, class Period, class F, class... Args>
auto ThreadPool::submit_with_timeout_priority(
    int priority,
    const std::chrono::duration<Rep, Period>& timeout,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    CancellationSource source;

    source.cancel_after(timeout);


    return submit_cancellable_token(
        priority,
        source.token(),
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


template <class F, class... Args>
auto ThreadPool::submit_with_deadline(
    std::chrono::steady_clock::time_point deadline,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    return submit_with_deadline_priority(
        0,
        deadline,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


template <class Rep, class Period, class F, class... Args>
auto ThreadPool::submit_with_timeout(
    const std::chrono::duration<Rep, Period>& timeout,
    F&& f,
    Args&&... args
)
    -> TaskResult<
        std::invoke_result_t<F, Args...>
    >
{
    return submit_with_timeout_priority(
        0,
        timeout,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// try_submit_token
// ============================================================

template <class F, class... Args>
auto ThreadPool::try_submit_token(
    CancellationToken token,
    F&& f,
    Args&&... args
)
    -> std::optional<
        std::future<
            std::invoke_result_t<F, Args...>
        >
    >
{
    SubmitResult status =
        SubmitResult::ACCEPTED;


    auto result = do_submit_token(
        0,
        std::move(token),
        false,
        status,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(status != SubmitResult::ACCEPTED
        &&
        status != SubmitResult::CALLER_RUNS)
    {
        return std::nullopt;
    }


    return std::move(result.future);
}


// ============================================================
// submit_at_token / submit_after_token
// ============================================================

template <class F, class... Args>
auto ThreadPool::submit_at_token(
    std::chrono::steady_clock::time_point at,
    CancellationToken token,
    F&& f,
    Args&&... args
)
    -> std::future<
        std::invoke_result_t<F, Args...>
    >
{
    SubmitResult status =
        SubmitResult::ACCEPTED;


    auto result = do_schedule_at_token(
        0,
        std::move(token),
        true,
        status,
        at,
        std::forward<F>(f),
        std::forward<Args>(args)...
    );


    if(status == SubmitResult::REJECTED_STOPPED)
    {
        throw std::runtime_error(
            "ThreadPool already stopped"
        );
    }


    return std::move(result.future);
}


template <class Rep, class Period, class F, class... Args>
auto ThreadPool::submit_after_token(
    const std::chrono::duration<Rep, Period>& delay,
    CancellationToken token,
    F&& f,
    Args&&... args
)
    -> std::future<
        std::invoke_result_t<F, Args...>
    >
{
    return submit_at_token(
        std::chrono::steady_clock::now()
            + delay,
        std::move(token),
        std::forward<F>(f),
        std::forward<Args>(args)...
    );
}


// ============================================================
// then (5.0.0)
// ============================================================

template <class T, class F>
auto ThreadPool::then(
    std::future<T> future,
    F&& f
)
    -> std::future<
        typename threadpool_detail::ThenResult<
            T,
            F
        >::type
    >
{
    using R =
        typename threadpool_detail::ThenResult<
            T,
            F
        >::type;


    /*
     * Both captures are copyable: std::future is move-only, so it lives
     * behind a shared_ptr for the same reason when_all() does - the
     * packaged_task inside submit() requires a copyable callable.
     */
    auto predecessor =
        std::make_shared<std::future<T>>(
            std::move(future)
        );

    auto fn =
        std::decay_t<F>(
            std::forward<F>(f)
        );


    return submit(
        [predecessor, fn]() mutable -> R
        {
            if constexpr(std::is_void_v<T>)
            {
                predecessor->get();

                if constexpr(std::is_void_v<R>)
                    fn();
                else
                    return fn();
            }
            else
            {
                T value = predecessor->get();

                if constexpr(std::is_void_v<R>)
                    fn(std::move(value));
                else
                    return fn(std::move(value));
            }
        }
    );
}


// ============================================================
// submit_batch (5.0.0)
// ============================================================

template <class F>
auto ThreadPool::submit_batch_priority(
    int priority,
    std::size_t count,
    F&& f
)
    -> std::vector<
        std::future<
            std::invoke_result_t<F, std::size_t>
        >
    >
{
    using R =
        std::invoke_result_t<F, std::size_t>;


    std::vector<std::future<R>> out;

    out.reserve(count);


    for(std::size_t i = 0; i < count; ++i)
    {
        out.emplace_back(
            submit_priority(
                priority,
                f,
                i
            )
        );
    }


    return out;
}


template <class F>
auto ThreadPool::submit_batch(
    std::size_t count,
    F&& f
)
    -> std::vector<
        std::future<
            std::invoke_result_t<F, std::size_t>
        >
    >
{
    return submit_batch_priority(
        0,
        count,
        std::forward<F>(f)
    );
}


// ============================================================
// TaskGroup - structured concurrency (5.0.0)
// ============================================================

enum class GroupDestroyPolicy
{
    WAIT,               // Destructor waits for every group task
    CANCEL_AND_WAIT     // Destructor requests cancellation first
};


/**
 * Thrown by TaskGroup::rethrow_exceptions() when at least one task failed.
 *
 * All exception objects are retained so a caller can inspect or rethrow
 * each one individually; the first one is also available as the most
 * common "just rethrow" case.
 */
class TaskGroupError : public std::runtime_error
{
public:
    explicit TaskGroupError(
        std::vector<std::exception_ptr> errors
    )
        : std::runtime_error(
            "ThreadPool: task group finished with exceptions"
        )
        , errors_(std::move(errors))
    {
    }


    const std::vector<std::exception_ptr>& errors() const
    {
        return errors_;
    }


    std::size_t size() const
    {
        return errors_.size();
    }


private:
    std::vector<std::exception_ptr> errors_;
};


/**
 * A group of tasks with a shared cancellation domain and its own wait().
 *
 * Unlike ThreadPool::wait(), which drains the entire pool, wait() here
 * only waits for tasks submitted through this group. That makes it safe
 * to use a long-lived pool with several independent concurrent groups.
 *
 * Every task runs as a normal pool task whose wrapper:
 *
 * - checks the group token before entering user code,
 * - delivers a value or an exception to its own future,
 * - records exceptions for rethrow_exceptions(),
 * - decrements the group's pending count under all outcomes.
 *
 * The last point is why the group uses the guarded submission engine
 * with skip_on_cancel = false: with submit_token() a cancelled task
 * would be skipped entirely and the wrapper could never decrement
 * pending. The same engine attaches an on_abandon hook, so tasks that
 * are discarded by clear() / stop(IMMEDIATE) / handle.cancel() still
 * release their bookkeeping and fulfil their futures with TaskCancelled.
 *
 * The group token is installed in the execution context too: inside a
 * group task, ThreadPool::current_token() is the group's token.
 *
 * Note: wait()/wait_for()/wait_until() must be called from a thread that
 * is not a worker of this (or any) pool - blocking a worker while its
 * group still needs that worker is a deadlock. The destructor swallows
 * that diagnostic, so create groups on an external thread.
 */
class TaskGroup
{
public:
    explicit TaskGroup(
        ThreadPool& pool,
        GroupDestroyPolicy policy =
            GroupDestroyPolicy::WAIT
    )
        : pool_(pool)
        , policy_(policy)
        , state_(std::make_shared<State>())
    {
    }


    ~TaskGroup()
    {
        if(policy_
            == GroupDestroyPolicy::CANCEL_AND_WAIT)
        {
            request_cancel();
        }


        try
        {
            wait();
        }
        catch(...)
        {
            /*
             * A destructor must not throw. On a worker thread this is the
             * deadlock guard described above; the group state stays alive
             * through shared_ptr until its tasks finish.
             */
        }
    }


    TaskGroup(const TaskGroup&) = delete;

    TaskGroup& operator=(const TaskGroup&) = delete;

    TaskGroup(TaskGroup&&) = delete;

    TaskGroup& operator=(TaskGroup&&) = delete;


    template <class F, class... Args>
    auto submit(
        F&& f,
        Args&&... args
    )
        -> std::future<
            std::invoke_result_t<F, Args...>
        >
    {
        return submit_priority(
            0,
            std::forward<F>(f),
            std::forward<Args>(args)...
        );
    }


    template <class F, class... Args>
    auto submit_priority(
        int priority,
        F&& f,
        Args&&... args
    )
        -> std::future<
            std::invoke_result_t<F, Args...>
        >
    {
        using R =
            std::invoke_result_t<F, Args...>;

        using Callable =
            std::decay_t<F>;

        using ArgsTuple =
            std::tuple<std::decay_t<Args>...>;


        static_assert(
            std::is_invocable_v<
                Callable&,
                std::decay_t<Args>...
            >,
            "TaskGroup: callable is not invocable with the given arguments"
        );


        auto promise =
            std::make_shared<std::promise<R>>();

        auto future =
            promise->get_future();

        auto token =
            state_->source.token();


        {
            std::lock_guard<std::mutex> lock(
                state_->mtx
            );

            state_->source.throw_if_cancelled();

            ++state_->pending;
            ++state_->submitted;
        }


        /*
         * clear(), stop(IMMEDIATE), handle.cancel() and a rejected
         * submission all prevent the wrapper from ever running. Without
         * this hook the group's pending counter would never reach zero and
         * wait() would hang forever.
         *
         * The hook also fulfils the caller's promise with TaskCancelled:
         * a discarded group task must not leave a future that never
         * completes.
         */
        auto abandon =
            [group = state_, promise]()
            {
                std::exception_ptr ep =
                    std::make_exception_ptr(
                        TaskCancelled()
                    );


                try
                {
                    promise->set_exception(ep);
                }
                catch(...)
                {
                }


                {
                    std::lock_guard<std::mutex> lock(
                        group->mtx
                    );

                    ++group->failed;

                    group->errors.push_back(ep);
                }


                finish_one(group);
            };


        ThreadPool::SubmitResult status =
            ThreadPool::SubmitResult::ACCEPTED;


        try
        {
            (void)pool_.do_submit_token_ex(
                priority,
                token,
                false,
                abandon,
                true,
                status,
                [
                    group = state_,
                    promise,
                    token,
                    callable =
                        Callable(
                            std::forward<F>(f)
                        ),
                    bound =
                        ArgsTuple(
                            std::forward<Args>(args)...
                        )
                ]() mutable
                {
                    try
                    {
                        token.throw_if_cancelled();


                        if constexpr(std::is_void_v<R>)
                        {
                            std::apply(
                                std::move(callable),
                                std::move(bound)
                            );

                            promise->set_value();
                        }
                        else
                        {
                            promise->set_value(
                                std::apply(
                                    std::move(callable),
                                    std::move(bound)
                                )
                            );
                        }
                    }
                    catch(...)
                    {
                        std::exception_ptr ep =
                            std::current_exception();


                        try
                        {
                            promise->set_exception(ep);
                        }
                        catch(...)
                        {
                            /*
                             * A promise can only be fulfilled once; a
                             * broken group must not kill the worker.
                             */
                        }


                        {
                            std::lock_guard<std::mutex> lock(
                                group->mtx
                            );

                            ++group->failed;

                            group->errors.push_back(ep);
                        }
                    }


                    finish_one(group);
                }
            );
        }
        catch(...)
        {
            /*
             * An exception before enqueueing (allocation failure, stopped
             * pool) means no wrapper and no run_task(): release here.
             */
            abandon();

            throw;
        }


        if(status
            != ThreadPool::SubmitResult::ACCEPTED
            &&
            status
            != ThreadPool::SubmitResult::CALLER_RUNS)
        {
            abandon();

            throw std::runtime_error(
                "TaskGroup: task rejected by the pool"
            );
        }


        return future;
    }


    /** Request cancellation of every not-yet-started task in the group. */
    bool request_cancel()
    {
        return state_->source.request_cancel();
    }


    /*
     * Arm a group-wide deadline (5.1.0). Every task submitted before the
     * deadline shares the domain; tasks that have not started when it
     * expires complete with TaskCancelled, while running tasks can check
     * ThreadPool::current_token().
     *
     * Unlike wait_for() this does not block the caller and does not
     * abandon the group: wait() still drains every task's bookkeeping.
     */
    template <class Rep, class Period>
    void cancel_after(
        const std::chrono::duration<Rep, Period>& delay
    )
    {
        state_->source.cancel_after(delay);
    }


    void cancel_at(
        std::chrono::steady_clock::time_point at
    )
    {
        state_->source.cancel_at(at);
    }


    bool timed_out() const
    {
        return state_->source.timed_out();
    }


    bool is_cancelled() const
    {
        return state_->source.is_cancelled();
    }


    CancellationToken token() const
    {
        return state_->source.token();
    }


    void wait()
    {
        ensure_external_thread();


        std::unique_lock<std::mutex> lock(
            state_->mtx
        );

        state_->cv.wait(
            lock,
            [this]()
            {
                return state_->pending == 0;
            }
        );
    }


    template <class Rep, class Period>
    bool wait_for(
        const std::chrono::duration<
            Rep,
            Period
        >& timeout
    )
    {
        ensure_external_thread();


        std::unique_lock<std::mutex> lock(
            state_->mtx
        );

        return state_->cv.wait_for(
            lock,
            timeout,
            [this]()
            {
                return state_->pending == 0;
            }
        );
    }


    template <class Clock, class Duration>
    bool wait_until(
        const std::chrono::time_point<
            Clock,
            Duration
        >& deadline
    )
    {
        ensure_external_thread();


        std::unique_lock<std::mutex> lock(
            state_->mtx
        );

        return state_->cv.wait_until(
            lock,
            deadline,
            [this]()
            {
                return state_->pending == 0;
            }
        );
    }


    /** Rethrow all recorded task exceptions as one TaskGroupError. */
    void rethrow_exceptions()
    {
        std::vector<std::exception_ptr> errors;

        {
            std::lock_guard<std::mutex> lock(
                state_->mtx
            );

            errors = state_->errors;
        }


        if(!errors.empty())
            throw TaskGroupError(std::move(errors));
    }


    std::vector<std::exception_ptr> errors() const
    {
        std::lock_guard<std::mutex> lock(
            state_->mtx
        );

        return state_->errors;
    }


    std::size_t pending() const
    {
        std::lock_guard<std::mutex> lock(
            state_->mtx
        );

        return state_->pending;
    }


    std::size_t submitted() const
    {
        std::lock_guard<std::mutex> lock(
            state_->mtx
        );

        return state_->submitted;
    }


    std::size_t failed() const
    {
        std::lock_guard<std::mutex> lock(
            state_->mtx
        );

        return state_->failed;
    }


private:
    struct State
    {
        mutable std::mutex mtx;

        std::condition_variable cv;

        std::size_t pending = 0;

        std::size_t submitted = 0;

        std::size_t failed = 0;

        std::vector<std::exception_ptr> errors;

        CancellationSource source;
    };


    static void finish_one(
        const std::shared_ptr<State>& state
    )
    {
        std::lock_guard<std::mutex> lock(
            state->mtx
        );

        --state->pending;

        if(state->pending == 0)
            state->cv.notify_all();
    }


    static void ensure_external_thread()
    {
        if(ThreadPool::current_worker_id()
            != ThreadPool::no_worker)
        {
            throw std::logic_error(
                "TaskGroup::wait() called from a worker thread"
            );
        }
    }


private:
    ThreadPool& pool_;

    GroupDestroyPolicy policy_;

    std::shared_ptr<State> state_;
};


#endif
