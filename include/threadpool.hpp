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
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>


enum class PoolStopMode
{
    GRACEFUL,   // Finish all remaining queued tasks before stopping
    IMMEDIATE   // Discard tasks that have not started yet
};


enum class RejectPolicy
{
    BLOCK,      // Block submit() when the queue is full
    DISCARD     // Reject the task immediately when the queue is full
};


enum class TaskState
{
    WAITING,
    RUNNING,
    COMPLETED,
    CANCELLED
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
        TaskState expected = TaskState::WAITING;

        return state_->compare_exchange_strong(
            expected,
            TaskState::CANCELLED,
            std::memory_order_acq_rel
        );
    }

    bool is_cancelled() const
    {
        return state_->load(std::memory_order_acquire)
            == TaskState::CANCELLED;
    }

    TaskState state() const
    {
        return state_->load(std::memory_order_acquire);
    }

private:
    explicit TaskHandle(
        std::shared_ptr<std::atomic<TaskState>> state)
        : state_(std::move(state))
    {
    }

private:
    std::shared_ptr<std::atomic<TaskState>> state_ =
        std::make_shared<std::atomic<TaskState>>(
            TaskState::WAITING
        );

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

        RejectPolicy reject = RejectPolicy::BLOCK;
    };


public:

    explicit ThreadPool(Config cfg = Config{});

    ~ThreadPool();


    /**
     * Submit a normal task.
     *
     * Default priority = 0.
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


    void set_reject_callback(std::function<void()> cb);


    void stop(
        PoolStopMode mode = PoolStopMode::GRACEFUL
    );


    PoolMetrics get_metrics() const;


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


private:

    void worker_loop();

    void create_worker_locked();

    void maybe_expand_locked();


private:

    Config cfg_;


    std::priority_queue<
        Task,
        std::vector<Task>,
        TaskCompare
    > tasks_;


    /*
     * Exited std::thread objects cannot simply be removed before join().
     *
     * worker_count_ represents the number of currently active workers.
     */
    std::vector<std::thread> threads_;


    mutable std::mutex mtx_;

    std::condition_variable cv_task_;

    std::condition_variable cv_queue_;


    bool stopping_ = false;

    PoolStopMode stop_mode_ =
        PoolStopMode::GRACEFUL;


    std::size_t worker_count_ = 0;

    std::size_t idle_count_ = 0;


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
    using ReturnType =
        std::invoke_result_t<F, Args...>;


    /*
     * Bind the callable and its arguments into a zero-argument function.
     */
    auto bound =
        std::bind(
            std::forward<F>(f),
            std::forward<Args>(args)...
        );


    auto packaged =
        std::make_shared<
            std::packaged_task<ReturnType()>
        >(
            std::move(bound)
        );


    auto future = packaged->get_future();


    auto state =
        std::make_shared<
            std::atomic<TaskState>
        >(
            TaskState::WAITING
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
        [this, packaged, state]()
        {
            /*
             * Transition from WAITING to RUNNING.
             *
             * If cancel() succeeds first, the state will already be
             * CANCELLED and the worker will skip task execution.
             */
            TaskState expected =
                TaskState::WAITING;


            if(!state->compare_exchange_strong(
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
                (*packaged)();
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


            state->store(
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
        std::unique_lock<std::mutex> lock(mtx_);


        if(stopping_)
        {
            rejected_.fetch_add(
                1,
                std::memory_order_relaxed
            );

            throw std::runtime_error(
                "ThreadPool already stopped"
            );
        }


        /*
         * Handle a full task queue.
         */
        if(tasks_.size() >= cfg_.max_queue)
        {
            if(cfg_.reject == RejectPolicy::BLOCK)
            {
                cv_queue_.wait(
                    lock,
                    [this]()
                    {
                        return stopping_
                            ||
                            tasks_.size()
                            < cfg_.max_queue;
                    }
                );


                if(stopping_)
                {
                    rejected_.fetch_add(
                        1,
                        std::memory_order_relaxed
                    );

                    throw std::runtime_error(
                        "ThreadPool stopped while submit was waiting"
                    );
                }
            }
            else
            {
                rejected_.fetch_add(
                    1,
                    std::memory_order_relaxed
                );


                reject_callback =
                    reject_callback_;


                lock.unlock();


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


                throw std::runtime_error(
                    "ThreadPool queue full, task rejected"
                );
            }
        }


        tasks_.push(
            std::move(task)
        );


        submitted_.fetch_add(
            1,
            std::memory_order_relaxed
        );


        /*
         * Dynamically expand the worker pool according to the current load.
         */
        maybe_expand_locked();
    }


    cv_task_.notify_one();


    return TaskResult<ReturnType>{
        std::move(future),
        std::move(handle)
    };
}


#endif