#include "threadpool.hpp"


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
     *
     * The workers are moved out under the lock and joined without holding
     * it: a worker still needs the mutex before it can finish its last
     * iteration.
     */
    std::vector<
        std::unique_ptr<Worker>
    > pending;


    {
        std::lock_guard<std::mutex> lock(mtx_);

        pending.swap(workers_);
    }


    for(auto& worker : pending)
    {
        if(worker->thread.joinable())
        {
            worker->thread.join();
        }
    }
}


// ============================================================
// reap_workers_locked
// ============================================================

void ThreadPool::reap_workers_locked()
{
    /*
     * The caller must already hold mtx_.
     *
     * A Worker is removed only after the corresponding thread left
     * worker_loop(). Without this step, exited std::thread objects would
     * stay in workers_ forever and every expansion / shrink cycle would
     * add one more dead entry.
     *
     * Joining here is safe because a finished worker no longer touches
     * mtx_.
     */
    for(auto it = workers_.begin();
        it != workers_.end();)
    {
        Worker& worker = **it;


        if(worker.finished.load(
            std::memory_order_acquire))
        {
            if(worker.thread.joinable())
            {
                worker.thread.join();
            }

            it = workers_.erase(it);

            --exited_pending_;
        }
        else
        {
            ++it;
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
     * reap_workers_locked() is driven by maybe_expand_locked() so that a
     * submit pays for it only when a worker actually exited.
     */
    workers_.push_back(
        std::make_unique<Worker>()
    );


    Worker* worker = workers_.back().get();


    /*
     * The order here is important:
     *
     * increment worker_count_ before starting the thread.
     *
     * Otherwise, a newly created worker could start and exit immediately,
     * creating a race in the worker counter.
     */
    ++worker_count_;

    ++workers_created_;


    try
    {
        worker->thread =
            std::thread(
                &ThreadPool::worker_loop,
                this,
                worker
            );
    }
    catch(...)
    {
        --worker_count_;

        --workers_created_;

        workers_.pop_back();

        throw;
    }
}


// ============================================================
// run_dropped
// ============================================================

void ThreadPool::run_dropped(
    std::vector<Task>& dropped
)
{
    /*
     * Runs outside of mtx_.
     *
     * The wrapper detects that the state is no longer WAITING, counts the
     * cancellation and breaks the promise, which would otherwise wait
     * forever.
     */
    for(auto& task : dropped)
    {
        run_task(task);
    }


    dropped.clear();
}


// ============================================================
// mark_exit_locked
// ============================================================

void ThreadPool::mark_exit_locked()
{
    /*
     * The caller must already hold mtx_.
     *
     * worker_count_ describes the workers that are still able to run
     * tasks, so it drops immediately.
     *
     * exited_pending_ keeps the difference to workers_ visible, which
     * tells maybe_expand_locked() that there is a thread to join.
     */
    --worker_count_;

    ++exited_pending_;
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

    /*
     * Reclaim exited workers first.
     *
     * Doing this before the max_threads check is what keeps workers_ from
     * filling up with dead entries when the pool never grows again.
     */
    if(exited_pending_ > 0)
        reap_workers_locked();


    if(worker_count_ >= cfg_.max_threads)
        return;


    if(tasks_.empty())
        return;


    if(tasks_.size() <= idle_count_)
        return;


    create_worker_locked();
}


// ============================================================
// pop_task_locked
// ============================================================

bool ThreadPool::pop_task_locked(Task& out)
{
    /*
     * The caller must already hold mtx_.
     */

    if(tasks_.empty())
        return false;


    /*
     * priority_queue exposes only a const reference to its top element.
     *
     * The element itself is not const, so moving it out through
     * const_cast is well defined here.
     */
    out =
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


    return true;
}


// ============================================================
// discard_tasks_locked
// ============================================================

std::size_t ThreadPool::discard_tasks_locked(
    std::vector<Task>& dropped
)
{
    /*
     * The caller must already hold mtx_.
     *
     * Every queued task is marked CANCELLED so that no worker can start
     * running it later.
     *
     * The tasks themselves go back to the caller, which has to pass them
     * to run_task() once the lock is released: that is what breaks their
     * promises and counts them as cancelled. Doing it under the lock would
     * run user-side destructors while holding mtx_.
     */
    dropped.clear();


    Task task;


    while(pop_task_locked(task))
    {
        if(task.state)
        {
            TaskState expected =
                TaskState::WAITING;


            task.state->compare_exchange_strong(
                expected,
                TaskState::CANCELLED,
                std::memory_order_acq_rel
            );
        }


        dropped.emplace_back(
            std::move(task)
        );


        task = Task{};
    }


    return dropped.size();
}


// ============================================================
// enqueue_task
// ============================================================

ThreadPool::SubmitResult ThreadPool::enqueue_task(
    Task& task,
    bool allow_wait,
    std::function<void()>& reject_cb
)
{
    std::unique_lock<std::mutex> lock(mtx_);


    if(stopping_)
    {
        rejected_.fetch_add(
            1,
            std::memory_order_relaxed
        );


        reject_cb = reject_callback_;


        return SubmitResult::REJECTED_STOPPED;
    }


    /*
     * Handle a full task queue.
     */
    if(tasks_.size() >= cfg_.max_queue)
    {
        if(cfg_.reject
            == RejectPolicy::CALLER_RUNS)
        {
            /*
             * The caller executes the task itself.
             *
             * Nothing is pushed here, so the caller must keep the Task
             * object alive.
             */
            submitted_.fetch_add(
                1,
                std::memory_order_relaxed
            );


            return SubmitResult::CALLER_RUNS;
        }


        if(allow_wait
            &&
            cfg_.reject == RejectPolicy::BLOCK)
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


                reject_cb = reject_callback_;


                return SubmitResult::REJECTED_STOPPED;
            }
        }
        else
        {
            rejected_.fetch_add(
                1,
                std::memory_order_relaxed
            );


            reject_cb = reject_callback_;


            return SubmitResult::REJECTED_FULL;
        }
    }


    tasks_.push(
        std::move(task)
    );


    submitted_.fetch_add(
        1,
        std::memory_order_relaxed
    );


    if(tasks_.size() > peak_queue_)
    {
        peak_queue_ =
            tasks_.size();
    }


    /*
     * Dynamically expand the worker pool according to the current load.
     */
    maybe_expand_locked();


    /*
     * Wake one worker, but only if one is actually waiting.
     *
     * When every worker is busy executing a task, it revisits the queue as
     * soon as it finishes, so a notify would be a pointless syscall -
     * several hundred thousand of them per second under load.
     */
    if(idle_count_ > 0)
    {
        cv_task_.notify_one();
    }


    return SubmitResult::ACCEPTED;
}


// ============================================================
// worker_loop
// ============================================================

void ThreadPool::worker_loop(Worker* self)
{
    /*
     * Optional per-thread initialization hook.
     *
     * It runs without any internal lock held, exactly like a user task.
     */
    if(cfg_.thread_init)
    {
        try
        {
            cfg_.thread_init();
        }
        catch(...)
        {
            /*
             * A failing hook must not kill the worker.
             */
        }
    }


    /*
     * Reused between iterations so that batched pops rarely allocate.
     */
    std::vector<Task> batch;


    for(;;)
    {
        Task task;


        {
            std::unique_lock<std::mutex> lock(mtx_);


            ++idle_count_;


            bool timed_out = false;


            if(paused_)
            {
                /*
                 * Wait without a timeout while paused.
                 *
                 * Treating a pause as an idle timeout would shrink the
                 * pool even though the load did not change.
                 */
                cv_task_.wait(
                    lock,
                    [this]()
                    {
                        return stopping_
                            ||
                            !paused_;
                    }
                );
            }
            else
            {
                timed_out =
                    !cv_task_.wait_for(
                        lock,
                        cfg_.idle_timeout,
                        [this]()
                        {
                            return stopping_
                                ||
                                (!paused_
                                    &&
                                    !tasks_.empty());
                        }
                    );
            }


            --idle_count_;


            /*
             * Handle idle timeout.
             */
            if(timed_out)
            {
                /*
                 * Dynamically shrink the pool.
                 *
                 * Only workers above min_threads are allowed to exit.
                 */
                if(worker_count_ >
                    cfg_.min_threads)
                {
                    mark_exit_locked();

                    break;
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
                mark_exit_locked();

                break;
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
                mark_exit_locked();

                break;
            }


            /*
             * Resumed from a pause: re-evaluate the whole state instead of
             * assuming the queue has work for this worker.
             */
            if(paused_)
                continue;


            /*
             * Normally, if the condition-variable predicate succeeds,
             * tasks_ should not be empty.
             *
             * Keep this check for defensive robustness.
             */
            if(!pop_task_locked(task))
                continue;


            /*
             * Take a few extra tasks while the lock is held.
             *
             * One critical section serving several tiny tasks costs far
             * less than one acquisition per task.
             *
             * batch_size > 1 does not change ordering guarantees seen by
             * the callers: the tasks are still executed one after another.
             */
            batch.clear();


            const std::size_t limit =
                cfg_.batch_size > 0
                    ? cfg_.batch_size
                    : 1;


            for(std::size_t i = 1;
                i < limit;
                ++i)
            {
                Task extra;


                if(!pop_task_locked(extra))
                    break;


                batch.emplace_back(
                    std::move(extra)
                );
            }


            /*
             * Count every taken task as active before releasing the lock.
             *
             * Otherwise wait() could return while a task has been taken
             * from the queue but has not started running yet.
             */
            inflight_ +=
                1 + batch.size();
        }


        /*
         * Execute the tasks outside the mutex.
         *
         * This is important.
         *
         * Holding the mutex while executing a user task would block
         * submit(), get_metrics(), stop(), and other workers.
         */
        run_task(task);


        for(auto& extra : batch)
        {
            run_task(extra);
        }


        {
            const std::size_t finished =
                1 + batch.size();


            /*
             * Clear before locking: destructors of captured objects must
             * not run under the mutex.
             */
            batch.clear();


            std::lock_guard<std::mutex> lock(mtx_);


            inflight_ -= finished;


            if(inflight_ == 0
                &&
                tasks_.empty())
            {
                /*
                 * Wake every thread waiting in wait() / wait_for().
                 */
                cv_idle_.notify_all();
            }
        }
    }


    /*
     * The worker must not lock mtx_ at this point.
     *
     * reap_workers_locked() joins finished workers while holding it, so
     * taking the mutex here would deadlock against the reaper.
     */
    self->finished.store(
        true,
        std::memory_order_release
    );
}


// ============================================================
// run_task
// ============================================================

void ThreadPool::run_task(Task& task)
{
    /*
     * Runs outside of mtx_.
     *
     * All bookkeeping lives inside the wrapper lambda created by
     * do_submit(), so this only has to make sure that neither an escaping
     * exception nor a captured destructor can kill the caller.
     */
    try
    {
        task.func();
    }
    catch(...)
    {
        /*
         * A single task must not terminate the worker thread or the
         * submitting thread.
         */
    }


    /*
     * Drop the user callable before the caller locks again.
     */
    task = Task{};
}


// ============================================================
// wait
// ============================================================

void ThreadPool::wait()
{
    std::unique_lock<std::mutex> lock(mtx_);


    cv_idle_.wait(
        lock,
        [this]()
        {
            return tasks_.empty()
                &&
                inflight_ == 0;
        }
    );
}


// ============================================================
// pause / resume
// ============================================================

void ThreadPool::pause()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);

        paused_ = true;
    }


    cv_task_.notify_all();
}


void ThreadPool::resume()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);

        paused_ = false;
    }


    cv_task_.notify_all();
}


bool ThreadPool::is_paused() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    return paused_;
}


// ============================================================
// clear
// ============================================================

std::size_t ThreadPool::clear()
{
    std::vector<Task> dropped;


    {
        std::lock_guard<std::mutex> lock(mtx_);

        discard_tasks_locked(dropped);
    }


    /*
     * run_dropped() empties the vector, so the count is taken first.
     */
    const std::size_t dropped_count =
        dropped.size();


    run_dropped(dropped);


    /*
     * A popped task may have released queue slots.
     */
    cv_queue_.notify_all();


    /*
     * The queue is empty now: wake threads waiting on drain.
     */
    cv_idle_.notify_all();


    return dropped_count;
}


// ============================================================
// stop
// ============================================================

void ThreadPool::stop(
    PoolStopMode mode
)
{
    std::vector<Task> dropped;


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


                discard_tasks_locked(dropped);
            }


            /*
             * Leaving the pool paused would prevent the workers that are
             * still executing tasks from draining the queue.
             */
            paused_ = false;


            cv_task_.notify_all();

            cv_queue_.notify_all();

            cv_idle_.notify_all();


            run_dropped(dropped);

            return;
        }


        stopping_ = true;

        stop_mode_ = mode;

        paused_ = false;


        if(mode == PoolStopMode::IMMEDIATE)
        {
            /*
             * Discard all tasks that have not started yet.
             */
            discard_tasks_locked(dropped);
        }
    }


    run_dropped(dropped);


    /*
     * Wake all worker threads.
     */
    cv_task_.notify_all();


    /*
     * Wake all producers that may be blocked in submit().
     */
    cv_queue_.notify_all();


    /*
     * The pool can no longer accept work, so drain waiters are released as
     * soon as the running tasks finish.
     */
    cv_idle_.notify_all();
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

        metrics.min_threads =
            cfg_.min_threads;

        metrics.max_threads =
            cfg_.max_threads;

        metrics.active =
            inflight_ + tasks_.size();

        metrics.peak_queue =
            peak_queue_;

        metrics.workers_created =
            workers_created_;

        metrics.slots =
            workers_.size();
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
// get_config
// ============================================================

ThreadPool::Config ThreadPool::get_config() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    return cfg_;
}


// ============================================================
// is_stopped
// ============================================================

bool ThreadPool::is_stopped() const
{
    std::lock_guard<std::mutex> lock(mtx_);

    return stopping_;
}
