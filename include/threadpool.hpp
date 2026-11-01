#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

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
 */
#define THREADPOOL_VERSION_MAJOR 3
#define THREADPOOL_VERSION_MINOR 0
#define THREADPOOL_VERSION_PATCH 0

#define THREADPOOL_VERSION_STRING "3.0.0"


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
}


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
     * Tasks currently held for a future execution time, i.e. deferred by
     * submit_after() / submit_at(). They are not yet in the runnable queue.
     */
    std::size_t scheduled = 0;
};


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


private:

    struct Task
    {
        int priority = 0;

        std::uint64_t sequence = 0;

        std::shared_ptr<std::atomic<TaskState>> state;

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


    using TaskQueue =
        std::priority_queue<
            Task,
            std::vector<Task>,
            TaskCompare
        >;


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
    struct Worker
    {
        std::thread thread;

        std::atomic<bool> finished{false};

        std::mutex local_mtx;

        std::deque<Task> local;
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

    void create_worker_locked();

    void maybe_expand_locked();

    void reap_workers_locked();


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
     * Entries are removed by reap_workers_locked() once a worker exited.
     */
    std::vector<
        std::unique_ptr<Worker>
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
    std::size_t inflight_ = 0;


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
                inflight_ == 0
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
                inflight_ == 0
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
     */
    auto body =
        std::make_shared<Body>(
            [
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
    task.func =
        [this, body]()
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


            state_ref.store(
                TaskState::COMPLETED,
                std::memory_order_release
            );


            completed_.fetch_add(
                1,
                std::memory_order_relaxed
            );
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
// do_schedule_at
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


    auto body =
        std::make_shared<Body>(
            [
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


    task.func =
        [this, body]()
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


            state_ref.store(
                TaskState::COMPLETED,
                std::memory_order_release
            );


            completed_.fetch_add(
                1,
                std::memory_order_relaxed
            );
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


#endif
