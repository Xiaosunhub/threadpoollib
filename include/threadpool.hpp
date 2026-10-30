#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
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
 * wait_until() and the expanded PoolMetrics only exist from 2.1.0 on.
 */
#define THREADPOOL_VERSION_MAJOR 2
#define THREADPOOL_VERSION_MINOR 1
#define THREADPOOL_VERSION_PATCH 1

#define THREADPOOL_VERSION_STRING "2.1.1"


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


private:

    struct Task
    {
        int priority = 0;

        std::uint64_t sequence = 0;

        std::shared_ptr<std::atomic<TaskState>> state;

        std::function<void()> func;
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
     */
    struct Worker
    {
        std::thread thread;

        std::atomic<bool> finished{false};
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
                inflight_ == 0;
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
                inflight_ == 0;
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


#endif
