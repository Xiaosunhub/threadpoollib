#include "threadpool.hpp"

#include <string>


namespace
{
    /*
     * The pool whose worker the current thread is, or nullptr.
     *
     * Thread local: a thread can only be a worker of one pool at a time,
     * while a process may hold several pools at once.
     */
    thread_local const ThreadPool* t_worker_of = nullptr;


    /*
     * The Worker* (stored as void*) of the worker the current thread is, or
     * nullptr for a non-worker thread.
     *
     * Lets enqueue_task() route a submission made from inside a task back to
     * that worker's own local queue instead of the shared global queue.
     */
    thread_local void* t_worker_self = nullptr;


    /*
     * 5.0.0 execution context.
     *
     * run_task() swaps t_current_token for the duration of one task, so a
     * callable can reach its own cancellation domain (and its worker id)
     * without any argument threading. The worker id is assigned once per
     * worker thread by WorkerScope, not per task.
     */
    thread_local CancellationToken t_current_token;

    thread_local std::size_t t_worker_id =
        ThreadPool::no_worker;

    std::atomic<std::size_t> g_next_worker_id{1};


    /*
     * Marks the thread as a worker for the lifetime of the object.
     *
     * RAII rather than a manual reset so that the flag is also cleared on
     * an early return. The previous context is saved because a worker may
     * synchronously run a task of another pool (CALLER_RUNS), and that
     * nested context must be restored afterwards.
     */
    struct WorkerScope
    {
        explicit WorkerScope(
            const ThreadPool* pool,
            void* self = nullptr
        )
            : previous_(t_worker_of)
            , prev_self_(t_worker_self)
            , prev_token_(t_current_token)
            , prev_id_(t_worker_id)
        {
            t_worker_of = pool;
            t_worker_self = self;

            t_current_token = CancellationToken{};

            if(pool)
            {
                t_worker_id =
                    g_next_worker_id.fetch_add(
                        1,
                        std::memory_order_relaxed
                    );
            }
            else
            {
                t_worker_id = ThreadPool::no_worker;
            }
        }

        ~WorkerScope()
        {
            t_worker_of = previous_;
            t_worker_self = prev_self_;
            t_current_token = prev_token_;
            t_worker_id = prev_id_;
        }

        WorkerScope(const WorkerScope&) = delete;

        WorkerScope& operator=(const WorkerScope&) = delete;

    private:
        const ThreadPool* previous_;
        void* prev_self_;
        CancellationToken prev_token_;
        std::size_t prev_id_;
    };
}


// ============================================================
// task execution context (5.0.0)
// ============================================================

std::size_t ThreadPool::current_worker_id()
{
    return t_worker_id;
}


const ThreadPool* ThreadPool::current_pool()
{
    return t_worker_of;
}


CancellationToken ThreadPool::current_token()
{
    return t_current_token;
}


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


    std::vector<Worker*> created;


    {
        std::lock_guard<std::mutex> lock(mtx_);


        /*
         * Cap the ready-queue backing store at max_queue up front so that
         * pushing under the lock never triggers a reallocation that copies
         * every surviving element while the pool lock is held.
         */
        tasks_.reserve(cfg_.max_queue);


        /*
         * Create the minimum number of resident worker threads.
         */
        for(std::size_t i = 0;
            i < cfg_.min_threads;
            ++i)
        {
            Worker* w = create_worker_locked();

            if(w)
                created.push_back(w);
        }
    }


    /*
     * Start the OS threads outside of any lock.
     *
     * A failing spawn (e.g. the system ran out of threads) rolls the
     * corresponding Worker back under the lock; the others already run.
     */
    for(Worker* w : created)
    {
        try
        {
            spawn_worker(w);
        }
        catch(...)
        {
            std::lock_guard<std::mutex> lock(mtx_);

            rollback_worker_locked(w);

            throw;
        }
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
        std::shared_ptr<Worker>
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

void ThreadPool::reap_workers_locked(
    std::vector<std::shared_ptr<Worker>>& out
)
{
    /*
     * The caller must already hold mtx_.
     *
     * A Worker is removed only after the corresponding thread left
     * worker_loop(). Without this step, exited std::thread objects would
     * stay in workers_ forever and every expansion / shrink cycle would
     * add one more dead entry.
     *
     * The thread is NOT joined here: join() can block for milliseconds and
     * must not run while the pool lock is held (it would stall every other
     * submitter and every worker waiting on mtx_). The caller joins the
     * collected Workers outside of the lock.
     */
    for(auto it = workers_.begin();
        it != workers_.end();)
    {
        Worker& worker = **it;


        if(worker.finished.load(
            std::memory_order_acquire))
        {
            out.push_back(
                std::move(*it)
            );

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
// spawn_worker
// ============================================================

void ThreadPool::spawn_worker(
    Worker* worker
)
{
    /*
     * Called WITHOUT mtx_ held.
     *
     * Moving the OS-thread creation out of the pool lock keeps a burst of
     * expansions from serializing every other submit on the (sometimes
     * multi-millisecond) cost of spawning a thread.
     */
    worker->thread =
        std::thread(
            &ThreadPool::worker_loop,
            this,
            worker
        );
}


// ============================================================
// rollback_worker_locked
// ============================================================

void ThreadPool::rollback_worker_locked(
    Worker* worker
)
{
    /*
     * The caller must already hold mtx_.
     *
     * Undoes create_worker_locked() when spawn_worker() failed, so the
     * counters stay consistent and the half-built Worker is reclaimed.
     */
    for(auto it = workers_.begin();
        it != workers_.end();
        ++it)
    {
        if(it->get() == worker)
        {
            workers_.erase(it);

            break;
        }
    }


    --worker_count_;

    --workers_created_;
}


// ============================================================
// create_worker_locked
// ============================================================

ThreadPool::Worker* ThreadPool::create_worker_locked()
{
    /*
     * The caller must already hold mtx_.
     *
     * Only builds the Worker object and updates the counters. The OS thread
     * is started afterwards by spawn_worker() outside of the lock.
     */

    if(stopping_)
        return nullptr;


    if(worker_count_ >= cfg_.max_threads)
        return nullptr;


        workers_.push_back(
            std::make_shared<Worker>()
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


    return worker;
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

ThreadPool::Worker* ThreadPool::maybe_expand_locked(
    std::vector<std::shared_ptr<Worker>>& out
)
{
    /*
     * The caller must already hold mtx_.
     *
     * Expansion conditions:
     *
     * 1. There are queued tasks.
     * 2. The number of queued tasks exceeds the number of idle workers.
     * 3. The pool has not reached max_threads.
     *
     * Returns the newly created Worker (to be spawned outside the lock), or
     * nullptr. Exited workers are reaped into out for the caller to join
     * outside the lock.
     */

    /*
     * Reclaim exited workers first.
     *
     * Doing this before the max_threads check is what keeps workers_ from
     * filling up with dead entries when the pool never grows again.
     */
    if(exited_pending_ > 0)
        reap_workers_locked(out);


    if(worker_count_ >= cfg_.max_threads)
        return nullptr;


    if(tasks_.empty())
        return nullptr;


    if(tasks_.size() <= idle_count_)
        return nullptr;


    return create_worker_locked();
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
     * tasks_ is a max-heap on TaskCompare kept by push_heap / pop_heap.
     *
     * pop_heap() moves the largest element (highest priority, then
     * smallest sequence) to the back; we then move it out and shrink.
     */
    std::pop_heap(
        tasks_.begin(),
        tasks_.end(),
        TaskCompare{}
    );

    out = std::move(tasks_.back());

    tasks_.pop_back();


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


    /*
     * Deferred (scheduled_) tasks have not started either.
     *
     * They are keyed by deadline, so iterate and cancel each one.
     */
    for(auto& entry : scheduled_)
    {
        Task& t = entry.second;


        if(t.state)
        {
            TaskState expected =
                TaskState::WAITING;


            t.state->compare_exchange_strong(
                expected,
                TaskState::CANCELLED,
                std::memory_order_acq_rel
            );
        }


        dropped.emplace_back(
            std::move(t)
        );
    }

    scheduled_.clear();


    /*
     * Per-worker deques hold nested submissions that have not been picked up
     * yet (and any backlog an idle peer has not stolen).
     *
     * Each deque is owned by exactly one worker, so take that worker's deque
     * mutex while draining to avoid racing with a task that is mid-submit on
     * the same worker, or with a peer that is mid-steal.
     */
    for(auto& worker_ptr : workers_)
    {
        Worker& w = *worker_ptr;


        std::lock_guard<std::mutex> lk(
            w.deque_.mtx_
        );


        while(!w.deque_.q_.empty())
        {
            Task t =
                std::move(
                    w.deque_.q_.back()
                );

            w.deque_.q_.pop_back();


            local_pending_.fetch_sub(
                1,
                std::memory_order_relaxed
            );


            if(t.state)
            {
                TaskState expected =
                    TaskState::WAITING;


                t.state->compare_exchange_strong(
                    expected,
                    TaskState::CANCELLED,
                    std::memory_order_acq_rel
                );
            }


            dropped.emplace_back(
                std::move(t)
            );
        }
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
    /*
     * Route a submission made from inside one of our own tasks back to that
     * worker's local LIFO queue.
     *
     * This is the structural optimization behind 3.0.0: fork-join style
     * work (parallel_for, parallel_reduce, recursive fan-out) no longer
     * serializes on the single global mutex. The owning worker is the only
     * producer and the only consumer, so the contention is gone in the
     * common case.
     *
     * The pool is not marked stopping_ here: a still-running task may
     * legitimately hand off more work, and stop() drains local queues
     * itself.
     */
    if(t_worker_self != nullptr)
    {
        auto* self =
            static_cast<Worker*>(
                t_worker_self
            );


        {
            std::lock_guard<std::mutex> lk(
                self->deque_.mtx_
            );

            self->deque_.q_.push_back(
                std::move(task)
            );
        }


        local_pending_.fetch_add(
            1,
            std::memory_order_relaxed
        );

        submitted_.fetch_add(
            1,
            std::memory_order_relaxed
        );

        /*
         * An idle peer is parked in worker_loop(). Wake one so it can come
         * and steal the backlog we just created, instead of waiting out its
         * full idle timeout before noticing the work.
         */
        cv_task_.notify_one();

        return SubmitResult::ACCEPTED;
    }


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


    tasks_.push_back(
        std::move(task)
    );

    std::push_heap(
        tasks_.begin(),
        tasks_.end(),
        TaskCompare{}
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
     *
     * maybe_expand_locked() returns a freshly created Worker (to be started
     * outside the lock) and reaps exited ones into to_join.
     */
    std::vector<std::shared_ptr<Worker>> to_join;

    Worker* new_worker =
        maybe_expand_locked(to_join);


    /*
     * Decide whether a wakeup is warranted while we still hold the lock.
     *
     * Wake one worker only if one is actually waiting: when every worker
     * is busy executing a task, it revisits the queue as soon as it
     * finishes, so a notify would be a pointless syscall - several hundred
     * thousand of them per second under load.
     */
    const bool wake = (idle_count_ > 0);


    /*
     * Release the lock before any syscall or thread spawn.
     *
     * A thread that is woken while the mutex is still held would
     * immediately block again on it, which both wastes a context switch
     * and extends the critical section seen by every other submitter.
     */
    lock.unlock();


    /*
     * Join exited workers out here, never under mtx_: a join() can block
     * for milliseconds and would otherwise stall the whole pool.
     */
    for(auto& w : to_join)
    {
        if(w->thread.joinable())
            w->thread.join();
    }


    /*
     * Start the newly created worker out here as well.
     *
     * On a spawn failure, roll the Worker back under the lock; the task we
     * just queued is still drained by the remaining workers.
     */
    if(new_worker)
    {
        try
        {
            spawn_worker(new_worker);
        }
        catch(...)
        {
            /*
             * Named differently from the outer unique_lock: C4456 at /W4
             * otherwise complains about a shadowed local declaration.
             */
            std::lock_guard<std::mutex> rollback_lock(
                mtx_
            );

            rollback_worker_locked(new_worker);
        }
    }


    if(wake)
        cv_task_.notify_one();


    return SubmitResult::ACCEPTED;
}


// ============================================================
// promote_scheduled_locked
// ============================================================

void ThreadPool::promote_scheduled_locked()
{
    /*
     * The caller must already hold mtx_.
     *
     * Moves every deferred task whose deadline has passed into the
     * runnable priority queue, preserving its priority and sequence.
     *
     * Never called while paused_: the caller guards that, because a paused
     * pool must not start draining deferred work.
     */
    if(scheduled_.empty())
        return;


    const auto now =
        std::chrono::steady_clock::now();


    auto it = scheduled_.begin();


    while(it != scheduled_.end()
        &&
        it->first <= now)
    {
        tasks_.push_back(
            std::move(it->second)
        );

        std::push_heap(
            tasks_.begin(),
            tasks_.end(),
            TaskCompare{}
        );

        if(tasks_.size() > peak_queue_)
            peak_queue_ = tasks_.size();


        it = scheduled_.erase(it);
    }
}


// ============================================================
// next_scheduled_locked
// ============================================================

std::chrono::steady_clock::time_point
    ThreadPool::next_scheduled_locked() const
{
    /*
     * The caller must already hold mtx_.
     */
    if(scheduled_.empty())
        return std::chrono::steady_clock::time_point::max();


    return scheduled_.begin()->first;
}


// ============================================================
// has_due_scheduled_locked
// ============================================================

bool ThreadPool::has_due_scheduled_locked() const
{
    /*
     * The caller must already hold mtx_.
     */
    if(scheduled_.empty())
        return false;


    return scheduled_.begin()->first
        <= std::chrono::steady_clock::now();
}


// ============================================================
// worker_loop
// ============================================================

void ThreadPool::worker_loop(Worker* self)
{
    /*
     * Tell wait() / wait_for() / wait_until() that the current thread
     * belongs to this pool.
     *
     * Without it, a task that calls wait() would block forever while
     * holding its own in-flight slot, and the caller would see a hang with
     * no diagnostic at all.
     */
    WorkerScope scope(
        this,
        self
    );


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


        /*
         * 1. Own deque, LIFO fast path.
         *
         * Tasks submitted from inside a task running on this worker land in
         * self->deque_ instead of the global queue. The owner is the only
         * producer, so this runs lock-free in the steady state and keeps
         * fork-join patterns (parallel_for, parallel_reduce, recursive
         * fan-out) off the shared mutex.
         */
        {
            std::lock_guard<std::mutex> lk(
                self->deque_.mtx_
            );

            if(!self->deque_.q_.empty())
            {
                task =
                    std::move(
                        self->deque_.q_.back()
                    );

                self->deque_.q_.pop_back();
            }
        }

        if(task.func)
        {
            /*
             * Count it as in-flight BEFORE removing it from the deque
             * counter, so a waiter never observes a moment where the task is
             * in neither counter (which would let wait() return early).
             */
            inflight_.fetch_add(
                1,
                std::memory_order_relaxed
            );

            local_pending_.fetch_sub(
                1,
                std::memory_order_relaxed
            );

            run_task(task);

            complete_tasks(1);

            continue;
        }


        /*
         * 2. Steal from a peer's deque (FIFO end).
         *
         * This is the 4.0.0 load balancer: an idle worker that finds the
         * global queue empty pulls fork-join backlog off a busy peer
         * instead of going to sleep, so long tails do not stall on one
         * thread.
         */
        if(try_steal(task, self))
        {
            inflight_.fetch_add(
                1,
                std::memory_order_relaxed
            );

            local_pending_.fetch_sub(
                1,
                std::memory_order_relaxed
            );

            stolen_.fetch_add(
                1,
                std::memory_order_relaxed
            );

            run_task(task);

            complete_tasks(1);

            continue;
        }


        {
            std::unique_lock<std::mutex> lock(mtx_);


            ++idle_count_;


            bool timed_out = false;


            /*
             * Idle workers wait either for the idle-timeout or for the
             * next scheduled deadline, whichever comes first. Computing the
             * deadline lets a worker that is merely holding space for a
             * future task wake up exactly when that task becomes runnable.
             */
            auto deadline =
                std::chrono::steady_clock::time_point::max();


            if(!paused_)
            {
                deadline =
                    std::chrono::steady_clock::now()
                        + cfg_.idle_timeout;

                if(!scheduled_.empty())
                {
                    auto next =
                        scheduled_.begin()->first;

                    if(next < deadline)
                        deadline = next;
                }
            }


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
                    !cv_task_.wait_until(
                        lock,
                        deadline,
                        [this]()
                        {
                            return stopping_
                                ||
                                (!paused_
                                    &&
                                    (!tasks_.empty()
                                        ||
                                        has_due_scheduled_locked()
                                        ||
                                        local_pending_.load(
                                            std::memory_order_relaxed
                                        ) > 0));
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
                 * A pause that began while this worker was already waiting
                 * also makes wait_for() time out: its predicate is false
                 * for as long as paused_ holds.
                 *
                 * That timeout is not idleness. Shrinking here would drag
                 * the pool down to min_threads purely because it was
                 * paused, which is exactly what the paused branch above
                 * exists to prevent.
                 */
                if(paused_)
                    continue;


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
             * Exit only after all queued tasks AND all scheduled (not yet
             * due) tasks have been consumed.
             */
            if(stopping_
                &&
                tasks_.empty()
                &&
                scheduled_.empty())
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
             * Pull any deferred task whose deadline has arrived into the
             * runnable priority queue. Skipped while paused: a paused pool
             * must not start draining scheduled work.
             */
            promote_scheduled_locked();


            /*
             * The predicate may have woken us because a peer deque has
             * stealable work (local_pending_ > 0) rather than because the
             * global queue has tasks. In that case there is nothing to pop
             * here; loop back to the top and let the steal path take it.
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
             * Clear before releasing: destructors of captured objects must
             * not run under the mutex.
             */
            batch.clear();


            /*
             * Account for completion without the global mutex in the common
             * case.
             *
             * Only the transition of inflight_ to zero needs the lock: that
             * is the only moment wait()'s predicate can flip to "all done",
             * and the lock makes the check of the queues authoritative so a
             * wakeup is never lost.
             *
             * With the default batch_size of 1 this removes one mutex
             * acquire/release per completed task from the hot path.
             */
            complete_tasks(finished);
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
// try_steal
// ============================================================

bool ThreadPool::try_steal(
    Task& out,
    Worker* self
)
{
    /*
     * Called WITHOUT mtx_ held.
     *
     * Snapshot the worker list under mtx_ (as shared_ptr) so every
     * candidate Worker - and its deque_ mutex - stays alive for the whole
     * steal attempt, even if a peer is reaped and joined in the meantime.
     */
    std::vector<std::shared_ptr<Worker>> candidates;

    {
        std::lock_guard<std::mutex> lock(mtx_);

        candidates = workers_;
    }


    for(auto& wp : candidates)
    {
        Worker* w = wp.get();

        /*
         * Never steal from our own deque: that would fight the owner's LIFO
         * pop and is pointless (we would just take our own work).
         */
        if(w == self)
            continue;


        if(w->deque_.steal(out))
            return true;
    }


    return false;
}


// ============================================================
// complete_tasks
// ============================================================

void ThreadPool::complete_tasks(
    std::size_t n
)
{
    /*
     * Runs outside of mtx_ in the common case.
     *
     * The increment side (worker_loop) adds to inflight_ BEFORE removing
     * the task from its deque, so a waiter never sees a task in neither
     * counter. Dropping to zero is the only moment wait()'s predicate can
     * flip to "all done", and taking the lock there makes the queue check
     * authoritative so a wakeup is never lost.
     */
    const std::size_t now =
        inflight_.fetch_sub(
            n,
            std::memory_order_acq_rel
        ) - n;


    if(now == 0)
    {
        std::lock_guard<std::mutex> lock(mtx_);


        if(tasks_.empty()
            &&
            scheduled_.empty()
            &&
            local_pending_.load(
                std::memory_order_relaxed
            ) == 0)
        {
            cv_idle_.notify_all();
        }
    }
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
    /*
     * Install the task's cancellation domain in the thread-local context.
     *
     * Every execution path goes through run_task() - global queue, local
     * deque, stolen task, promoted scheduled task, caller-runs and dropped
     * tasks - so current_token() is always the token of the task actually
     * executing, never a stale one from an outer task.
     */
    /*
     * clear(), stop(IMMEDIATE) and TaskHandle::cancel() all mark a queued
     * task CANCELLED before its wrapper is entered. The wrapper's
     * WAITING -> RUNNING transition then fails and it returns without
     * invoking user code.
     *
     * Detect that case up front: a wrapper that owns external bookkeeping
     * (TaskGroup's pending counter and promise) must still release it, or
     * wait() would hang forever.
     *
     * A token cancellation is NOT an abandon: there the wrapper does run
     * and only the packaged_task callable is skipped, so the state starts
     * out WAITING here.
     */
    const bool abandoned_before_run =
        task.state
        &&
        task.state->load(
            std::memory_order_acquire
        ) == TaskState::CANCELLED;


    CancellationToken previous =
        t_current_token;

    t_current_token = task.token;


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


    t_current_token = previous;


    if(abandoned_before_run
        &&
        task.on_abandon)
    {
        try
        {
            task.on_abandon();
        }
        catch(...)
        {
            /*
             * A cleanup hook must not take the worker down either.
             */
        }
    }


    /*
     * Drop the user callable before the caller locks again.
     */
    task = Task{};
}


// ============================================================
// ensure_not_worker
// ============================================================

void ThreadPool::ensure_not_worker(
    const char* what
) const
{
    /*
     * Draining waits for inflight_ to reach zero, and the calling task is
     * itself counted in inflight_.
     *
     * A worker can therefore only be released if some other thread
     * finishes its work for it, which is not something the pool can
     * guarantee. Throwing is strictly better than hanging: the caller gets
     * a stack trace at the offending call site.
     */
    if(t_worker_of != this)
        return;


    throw std::logic_error(
        std::string("ThreadPool: ")
        + what
        + "() was called from a task running in the same pool"
    );
}


// ============================================================
// wait
// ============================================================

void ThreadPool::wait()
{
    ensure_not_worker("wait");


    std::unique_lock<std::mutex> lock(mtx_);


    cv_idle_.wait(
        lock,
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
        }
        else
        {
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
    }


    /*
     * Broken promises and captured destructors must not run under the
     * mutex: a captured object may call back into the pool.
     *
     * Both branches above funnel their dropped tasks here, so a repeated
     * stop() is handled exactly like the first one.
     */
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
            inflight_.load(
                std::memory_order_relaxed
            )
                + tasks_.size()
                + scheduled_.size()
                + local_pending_.load(
                    std::memory_order_relaxed
                );

        metrics.scheduled =
            scheduled_.size();

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

    metrics.stolen =
        stolen_.load(
            std::memory_order_relaxed
        );

    metrics.failed =
        failed_.load(
            std::memory_order_relaxed
        );

    metrics.token_cancelled =
        token_cancelled_.load(
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


// ============================================================
// resize
// ============================================================

void ThreadPool::resize(
    std::size_t min_threads,
    std::size_t max_threads
)
{
    if(min_threads == 0)
    {
        throw std::invalid_argument(
            "min_threads must be >= 1"
        );
    }


    if(max_threads < min_threads)
    {
        throw std::invalid_argument(
            "max_threads must be >= min_threads"
        );
    }


    std::vector<Worker*> created;

    std::vector<std::shared_ptr<Worker>> to_join;


    {
        std::lock_guard<std::mutex> lock(mtx_);


        cfg_.min_threads = min_threads;

        cfg_.max_threads = max_threads;


        /*
         * Reap exited workers now that they no longer fit the new floor.
         */
        if(exited_pending_ > 0)
            reap_workers_locked(to_join);


        /*
         * Grow toward the new floor right away.
         *
         * create_worker_locked() already honors stopping_ and the (possibly
         * raised) max_threads cap, so this cannot over-allocate. It only
         * builds the Worker objects; the threads are spawned below.
         */
        while(worker_count_ < cfg_.min_threads)
        {
            Worker* w = create_worker_locked();

            if(w)
                created.push_back(w);
        }
    }


    /*
     * Join exited workers outside of the lock.
     */
    for(auto& w : to_join)
    {
        if(w->thread.joinable())
            w->thread.join();
    }


    /*
     * Start the freshly created workers outside of the lock.
     */
    for(Worker* w : created)
    {
        try
        {
            spawn_worker(w);
        }
        catch(...)
        {
            std::lock_guard<std::mutex> lock(mtx_);

            rollback_worker_locked(w);
        }
    }


    /*
     * A lowered max_threads only bites as workers go idle.
     *
     * They are not interrupted mid-task: the existing idle-timeout shrink
     * path enforces worker_count_ > min_threads, so the pool simply settles
     * at the new min instead of bouncing back up to the old ceiling.
     */
}


// ============================================================
// set_idle_timeout
// ============================================================

void ThreadPool::set_idle_timeout(
    std::chrono::milliseconds timeout
)
{
    if(timeout.count() <= 0)
    {
        throw std::invalid_argument(
            "idle_timeout must be > 0"
        );
    }


    std::lock_guard<std::mutex> lock(mtx_);


    cfg_.idle_timeout = timeout;
}


// ============================================================
// set_reject_policy
// ============================================================

void ThreadPool::set_reject_policy(
    RejectPolicy policy
)
{
    std::lock_guard<std::mutex> lock(mtx_);


    cfg_.reject = policy;
}


// ============================================================
// set_batch_size
// ============================================================

void ThreadPool::set_batch_size(
    std::size_t batch_size
)
{
    std::lock_guard<std::mutex> lock(mtx_);


    /* Values below 1 are treated as 1 (see Config::batch_size). */
    cfg_.batch_size = batch_size < 1 ? 1 : batch_size;
}


// ============================================================
// queue_size
// ============================================================

std::size_t ThreadPool::queue_size() const
{
    std::lock_guard<std::mutex> lock(mtx_);


    return tasks_.size();
}


// ============================================================
// scheduled_count
// ============================================================

std::size_t ThreadPool::scheduled_count() const
{
    std::lock_guard<std::mutex> lock(mtx_);


    return scheduled_.size();
}


// ============================================================
// worker_count
// ============================================================

std::size_t ThreadPool::worker_count() const
{
    std::lock_guard<std::mutex> lock(mtx_);


    return workers_.size();
}


// ============================================================
// active_count
// ============================================================

std::size_t ThreadPool::active_count() const
{
    std::lock_guard<std::mutex> lock(mtx_);


    std::size_t local = 0;


    for(const auto& wp : workers_)
        local += wp->deque_.size();


    return tasks_.size()
        + local
        + scheduled_.size()
        + inflight_.load(
            std::memory_order_relaxed
        );
}
