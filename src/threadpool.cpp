#include "threadpool.hpp"

#include <algorithm>


ThreadPool::ThreadPool(Config cfg)
    : cfg_(std::move(cfg))
{
    if(cfg_.min_threads == 0)
    {
        throw std::invalid_argument(
            "min_threads must be >= 1"
        );
    }


    if(cfg_.max_threads < cfg_.min_threads)
    {
        throw std::invalid_argument(
            "max_threads must be >= min_threads"
        );
    }


    if(cfg_.max_queue == 0)
    {
        throw std::invalid_argument(
            "max_queue must be >= 1"
        );
    }


    if(cfg_.idle_timeout.count() <= 0)
    {
        throw std::invalid_argument(
            "idle_timeout must be > 0"
        );
    }


    std::lock_guard<std::mutex> lock(mtx_);


    /*
     * Create the minimum number of resident worker threads.
     */
    for(std::size_t i = 0;
        i < cfg_.min_threads;
        ++i)
    {
        create_worker_locked();
    }
}


ThreadPool::~ThreadPool()
{
    /*
     * Use graceful shutdown by default.
     */
    stop(PoolStopMode::GRACEFUL);


    /*
     * Worker threads must never be detached here.
     *
     * The destructor must wait until every worker has completely exited.
     * Otherwise, a worker could access this object after destruction,
     * causing a use-after-free bug.
     */
    for(auto& thread : threads_)
    {
        if(thread.joinable())
        {
            thread.join();
        }
    }
}


// ============================================================
// create_worker_locked
// ============================================================

void ThreadPool::create_worker_locked()
{
    /*
     * The caller must already hold mtx_.
     */

    if(stopping_)
        return;


    if(worker_count_ >= cfg_.max_threads)
        return;


    /*
     * The order here is important:
     *
     * increment worker_count_ before starting the thread.
     *
     * Otherwise, a newly created worker could start and exit immediately,
     * creating a race in the worker counter.
     */
    ++worker_count_;


    try
    {
        threads_.emplace_back(
            &ThreadPool::worker_loop,
            this
        );
    }
    catch(...)
    {
        --worker_count_;
        throw;
    }
}


// ============================================================
// maybe_expand_locked
// ============================================================

void ThreadPool::maybe_expand_locked()
{
    /*
     * The caller must already hold mtx_.
     *
     * Expansion conditions:
     *
     * 1. There are queued tasks.
     * 2. The number of queued tasks exceeds the number of idle workers.
     * 3. The pool has not reached max_threads.
     */

    if(worker_count_ >= cfg_.max_threads)
        return;


    if(tasks_.empty())
        return;


    if(tasks_.size() <= idle_count_)
        return;


    create_worker_locked();
}


// ============================================================
// worker_loop
// ============================================================

void ThreadPool::worker_loop()
{
    for(;;)
    {
        Task task;


        {
            std::unique_lock<std::mutex> lock(mtx_);


            ++idle_count_;


            bool awakened =
                cv_task_.wait_for(
                    lock,
                    cfg_.idle_timeout,
                    [this]()
                    {
                        return stopping_
                            ||
                            !tasks_.empty();
                    }
                );


            --idle_count_;


            /*
             * Handle idle timeout.
             */
            if(!awakened)
            {
                /*
                 * Dynamically shrink the pool.
                 *
                 * Only workers above min_threads are allowed to exit.
                 */
                if(worker_count_ >
                    cfg_.min_threads)
                {
                    --worker_count_;

                    return;
                }


                continue;
            }


            /*
             * IMMEDIATE shutdown:
             *
             * stop() has already discarded all queued tasks.
             * The worker exits immediately.
             */
            if(stopping_
                &&
                stop_mode_
                    == PoolStopMode::IMMEDIATE)
            {
                --worker_count_;

                return;
            }


            /*
             * GRACEFUL shutdown:
             *
             * Exit only after all queued tasks have been consumed.
             */
            if(stopping_
                &&
                tasks_.empty())
            {
                --worker_count_;

                return;
            }


            /*
             * Normally, if the condition-variable predicate succeeds,
             * tasks_ should not be empty.
             *
             * Keep this check for defensive robustness.
             */
            if(tasks_.empty())
            {
                continue;
            }


            task =
                std::move(
                    const_cast<Task&>(
                        tasks_.top()
                    )
                );


            tasks_.pop();


            /*
             * One queue slot is now available.
             * Wake one producer blocked in submit().
             */
            cv_queue_.notify_one();
        }


        /*
         * Execute the task outside the mutex.
         *
         * This is important.
         *
         * Holding the mutex while executing a user task would block
         * submit(), get_metrics(), stop(), and other workers.
         */
        try
        {
            task.func();
        }
        catch(...)
        {
            /*
             * A single task must not terminate the worker thread.
             */
        }
    }
}


// ============================================================
// stop
// ============================================================

void ThreadPool::stop(
    PoolStopMode mode
)
{
    {
        std::lock_guard<std::mutex> lock(mtx_);


        if(stopping_)
        {
            /*
             * A previous GRACEFUL shutdown request may be upgraded
             * to IMMEDIATE shutdown.
             */
            if(mode == PoolStopMode::IMMEDIATE
                &&
                stop_mode_
                    != PoolStopMode::IMMEDIATE)
            {
                stop_mode_ =
                    PoolStopMode::IMMEDIATE;


                while(!tasks_.empty())
                {
                    Task task =
                        std::move(
                            const_cast<Task&>(
                                tasks_.top()
                            )
                        );


                    tasks_.pop();


                    if(task.state)
                    {
                        TaskState expected =
                            TaskState::WAITING;


                        if(task.state
                            ->compare_exchange_strong(
                                expected,
                                TaskState::CANCELLED,
                                std::memory_order_acq_rel
                            ))
                        {
                            cancelled_.fetch_add(
                                1,
                                std::memory_order_relaxed
                            );
                        }
                    }
                }
            }


            cv_task_.notify_all();

            cv_queue_.notify_all();

            return;
        }


        stopping_ = true;

        stop_mode_ = mode;


        if(mode == PoolStopMode::IMMEDIATE)
        {
            /*
             * Discard all tasks that have not started yet.
             */
            while(!tasks_.empty())
            {
                Task task =
                    std::move(
                        const_cast<Task&>(
                            tasks_.top()
                        )
                    );


                tasks_.pop();


                if(task.state)
                {
                    TaskState expected =
                        TaskState::WAITING;


                    if(task.state
                        ->compare_exchange_strong(
                            expected,
                            TaskState::CANCELLED,
                            std::memory_order_acq_rel
                        ))
                    {
                        cancelled_.fetch_add(
                            1,
                            std::memory_order_relaxed
                        );
                    }
                }
            }
        }
    }


    /*
     * Wake all worker threads.
     */
    cv_task_.notify_all();


    /*
     * Wake all producers that may be blocked in submit().
     */
    cv_queue_.notify_all();
}


// ============================================================
// set_reject_callback
// ============================================================

void ThreadPool::set_reject_callback(
    std::function<void()> cb
)
{
    std::lock_guard<std::mutex> lock(mtx_);

    reject_callback_ = std::move(cb);
}


// ============================================================
// get_metrics
// ============================================================

PoolMetrics ThreadPool::get_metrics() const
{
    PoolMetrics metrics;


    {
        std::lock_guard<std::mutex> lock(mtx_);


        metrics.worker_count =
            worker_count_;

        metrics.idle_count =
            idle_count_;

        metrics.queue_size =
            tasks_.size();
    }


    metrics.submitted =
        submitted_.load(
            std::memory_order_relaxed
        );

    metrics.completed =
        completed_.load(
            std::memory_order_relaxed
        );

    metrics.cancelled =
        cancelled_.load(
            std::memory_order_relaxed
        );

    metrics.rejected =
        rejected_.load(
            std::memory_order_relaxed
        );

    metrics.running =
        running_.load(
            std::memory_order_relaxed
        );


    return metrics;
}


// ============================================================
// is_stopped
// ============================================================

bool ThreadPool::is_stopped() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    return stopping_;
}