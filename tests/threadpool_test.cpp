/*
 * Self-contained regression tests.
 *
 * Plain <cstdio> assertions instead of a test framework: the library must
 * stay usable without pulling in anything beyond the C++17 standard
 * library.
 *
 * Returns non-zero on the first failed expectation, so it can be used
 * directly as a CI check.
 */

#include "threadpool.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>


namespace
{
    int failures = 0;


    void expect(
        bool ok,
        const char* name
    )
    {
        std::printf(
            "%-52s %s\n",
            name,
            ok ? "ok" : "FAIL"
        );

        if(!ok)
            ++failures;
    }


    void sleep_ms(int ms)
    {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(ms)
        );
    }


    /*
     * Destructor calls back into the pool.
     *
     * stop() used to run dropped-task destructors while holding mtx_, so
     * this object deadlocked the pool on a non-recursive mutex.
     */
    struct ReentrantPayload
    {
        ThreadPool* pool = nullptr;

        ~ReentrantPayload()
        {
            if(pool)
            {
                /* Would block forever if mtx_ were still held. */
                const PoolMetrics m = pool->get_metrics();

                (void)m;
            }
        }
    };


    /*
     * Waits up to timeout_ms for pred() to become true.
     */
    template <class Pred>
    bool spin_until(
        Pred pred,
        int timeout_ms
    )
    {
        const int step = 10;

        for(int waited = 0;
            waited < timeout_ms;
            waited += step)
        {
            if(pred())
                return true;


            sleep_ms(step);
        }

        return pred();
    }


    ThreadPool::Config make_config()
    {
        ThreadPool::Config cfg;

        cfg.min_threads = 2;
        cfg.max_threads = 8;
        cfg.max_queue = 4096;

        cfg.idle_timeout =
            std::chrono::milliseconds(200);

        return cfg;
    }
}


// ------------------------------------------------------------
// Basic execution and argument forwarding
// ------------------------------------------------------------

static void test_basics()
{
    ThreadPool pool(make_config());


    auto sum =
        pool.submit(
            [](int a, int b)
            {
                return a + b;
            },
            20,
            22
        );


    expect(sum.get() == 42, "submit forwards arguments");


    /*
     * std::bind would have rejected this: the payload is move-only.
     */
    auto moved =
        pool.submit(
            [payload = std::make_unique<int>(99)]()
            {
                return *payload;
            }
        );


    expect(
        moved.get() == 99,
        "submit accepts move-only arguments"
    );


    auto voided =
        pool.submit(
            []()
            {
                /* void return */
            }
        );


    voided.wait();

    expect(true, "submit supports void tasks");


    auto boom =
        pool.submit(
            []() -> int
            {
                throw std::runtime_error("boom");
            }
        );


    bool caught = false;

    try
    {
        boom.get();
    }
    catch(const std::runtime_error&)
    {
        caught = true;
    }


    expect(caught, "task exceptions surface through the future");


    /* A throwing task must not have killed the worker. */
    auto alive = pool.submit([]() { return 5; });

    expect(alive.get() == 5, "pool survives a throwing task");
}


// ------------------------------------------------------------
// Priority ordering
// ------------------------------------------------------------

static void test_priority()
{
    ThreadPool::Config cfg = make_config();

    cfg.min_threads = 1;
    cfg.max_threads = 1;


    ThreadPool pool(cfg);


    std::promise<void> release;

    std::shared_future<void> gate =
        release.get_future().share();


    std::promise<void> started;

    auto blocker =
        pool.submit(
            [gate, &started]()
            {
                started.set_value();

                gate.wait();
            }
        );


    started.get_future().wait();


    std::mutex order_mtx;

    std::vector<int> order;


    for(int priority : {1, 5, 3})
    {
        pool.submit_priority(
            priority,
            [priority, &order, &order_mtx]()
            {
                std::lock_guard<std::mutex> lock(order_mtx);

                order.push_back(priority);
            }
        );
    }


    release.set_value();

    blocker.wait();

    pool.wait();


    expect(
        order.size() == 3,
        "all priority tasks ran"
    );

    expect(
        order.size() == 3
            &&
            order[0] == 5
            &&
            order[1] == 3
            &&
            order[2] == 1,
        "higher priority runs first"
    );
}


// ------------------------------------------------------------
// Cancellation
// ------------------------------------------------------------

static void test_cancel()
{
    ThreadPool::Config cfg = make_config();

    cfg.min_threads = 1;
    cfg.max_threads = 1;


    ThreadPool pool(cfg);


    std::promise<void> release;

    std::shared_future<void> gate =
        release.get_future().share();


    std::promise<void> started;

    auto blocker =
        pool.submit(
            [gate, &started]()
            {
                started.set_value();

                gate.wait();
            }
        );


    started.get_future().wait();


    auto result =
        pool.submit_cancellable(
            0,
            []()
            {
                return 1;
            }
        );


    expect(
        result.handle.valid(),
        "cancellable handle is valid"
    );

    expect(
        result.handle.cancel(),
        "cancel() succeeds while the task is WAITING"
    );

    expect(
        result.handle.is_cancelled(),
        "handle reports CANCELLED"
    );


    /* A second cancel() has nothing left to do. */
    expect(
        !result.handle.cancel(),
        "cancel() is idempotent"
    );


    release.set_value();

    blocker.wait();

    pool.wait();


    bool broken = false;

    try
    {
        result.future.get();
    }
    catch(const std::future_error&)
    {
        broken = true;
    }


    expect(
        broken,
        "cancelled task yields broken_promise, not a hang"
    );

    expect(
        pool.get_metrics().cancelled == 1,
        "cancel is counted in the metrics"
    );
}


// ------------------------------------------------------------
// Pause must not shrink the pool
// ------------------------------------------------------------

static void test_pause_keeps_workers()
{
    /*
     * A long idle timeout is what makes this deterministic.
     *
     * The bug needs a worker that is already idle-waiting when pause()
     * arrives: only then does it sit in wait_for() with a predicate that
     * stays false for as long as paused_ holds, and only then does the
     * resulting timeout get mistaken for idleness.
     *
     * With a short timeout the workers would shrink to min_threads before
     * the pause even starts, and the test would pass either way.
     */
    ThreadPool::Config cfg = make_config();

    cfg.idle_timeout =
        std::chrono::milliseconds(1500);


    ThreadPool pool(cfg);


    std::vector<std::future<void>> burst;


    for(int i = 0; i < 600; ++i)
    {
        burst.push_back(
            pool.submit(
                []()
                {
                    sleep_ms(1);
                }
            )
        );
    }


    const bool expanded =
        spin_until(
            [&pool]()
            {
                return pool.get_metrics().worker_count >= 4;
            },
            3000
        );


    expect(
        expanded,
        "pool expands under a burst"
    );


    /*
     * Drain, then pause while the workers are idle but still within their
     * idle timeout.
     */
    pool.wait_for(
        std::chrono::seconds(10)
    );


    const std::size_t before =
        pool.get_metrics().worker_count;


    pool.pause();


    /*
     * Longer than idle_timeout by a wide margin.
     *
     * Before the fix, a pause that began while the workers were already
     * waiting made wait_for() time out, and every worker above
     * min_threads exited.
     */
    sleep_ms(2200);


    const std::size_t during =
        pool.get_metrics().worker_count;


    pool.resume();


    pool.wait_for(
        std::chrono::seconds(10)
    );


    expect(
        during >= before
            &&
            before >= 4,
        "pause does not shrink the pool"
    );


    /* The burst must still have been completed in full. */
    std::size_t done = 0;

    for(auto& f : burst)
    {
        f.get();

        ++done;
    }


    expect(
        done == burst.size(),
        "every task completes after resume"
    );
}


// ------------------------------------------------------------
// Worker reclaim
// ------------------------------------------------------------

static void test_workers_are_reclaimed()
{
    ThreadPool pool(make_config());


    for(int round = 0; round < 6; ++round)
    {
        std::vector<std::future<void>> burst;


        for(int i = 0; i < 300; ++i)
        {
            burst.push_back(
                pool.submit(
                    []()
                    {
                        sleep_ms(1);
                    }
                )
            );
        }


        pool.wait_for(
            std::chrono::seconds(10)
        );


        for(auto& f : burst)
            f.get();


        sleep_ms(400);
    }


    /*
     * An idle worker only exits on its next wait cycle, and exited
     * threads are reclaimed on the next submission.
     */
    pool.submit([]() {}).get();

    pool.wait_for(
        std::chrono::seconds(5)
    );


    const PoolMetrics m = pool.get_metrics();


    expect(
        m.worker_count == m.min_threads,
        "workers shrink back to min_threads"
    );

    expect(
        m.slots == m.worker_count,
        "exited workers were reclaimed (no leaked slots)"
    );
}


// ------------------------------------------------------------
// wait() from inside the pool
// ------------------------------------------------------------

static void test_wait_from_worker()
{
    ThreadPool pool(make_config());


    auto inner =
        pool.submit(
            [&pool]()
            {
                try
                {
                    pool.wait();

                    return std::string("no-throw");
                }
                catch(const std::logic_error& e)
                {
                    return std::string(e.what());
                }
            }
        );


    expect(
        inner.get().find("same pool") != std::string::npos,
        "wait() inside a task throws logic_error"
    );


    auto timed =
        pool.submit(
            [&pool]()
            {
                try
                {
                    pool.wait_for(
                        std::chrono::milliseconds(1)
                    );

                    return 1;
                }
                catch(const std::logic_error&)
                {
                    return 2;
                }
            }
        );


    expect(
        timed.get() == 2,
        "wait_for() inside a task throws logic_error"
    );


    auto absolute =
        pool.submit(
            [&pool]()
            {
                try
                {
                    pool.wait_until(
                        std::chrono::steady_clock::now()
                    );

                    return 1;
                }
                catch(const std::logic_error&)
                {
                    return 2;
                }
            }
        );


    expect(
        absolute.get() == 2,
        "wait_until() inside a task throws logic_error"
    );


    auto after = pool.submit([]() { return 7; });

    expect(
        after.get() == 7,
        "pool still works after a rejected wait()"
    );
}


// ------------------------------------------------------------
// Default TaskHandle
// ------------------------------------------------------------

static void test_default_handle()
{
    TaskHandle handle;

    expect(
        !handle.valid(),
        "default TaskHandle is not valid"
    );

    expect(
        !handle.cancel(),
        "cancel() on a default handle is a no-op"
    );

    expect(
        handle.state() == TaskState::WAITING,
        "state() on a default handle is WAITING"
    );
}


// ------------------------------------------------------------
// Shutdown paths
// ------------------------------------------------------------

static void test_shutdown()
{
    ThreadPool::Config cfg = make_config();

    cfg.min_threads = 1;
    cfg.max_threads = 1;


    ThreadPool pool(cfg);


    std::promise<void> release;

    std::shared_future<void> gate =
        release.get_future().share();


    std::promise<void> started;

    auto blocker =
        pool.submit(
            [gate, &started]()
            {
                started.set_value();

                gate.wait();
            }
        );


    started.get_future().wait();


    for(int i = 0; i < 8; ++i)
    {
        auto payload =
            std::make_shared<ReentrantPayload>();

        payload->pool = &pool;


        pool.try_submit(
            [payload]()
            {
                (void)payload;
            }
        );
    }


    /* GRACEFUL first, then upgraded to IMMEDIATE. */
    pool.stop(PoolStopMode::GRACEFUL);

    pool.stop(PoolStopMode::IMMEDIATE);


    expect(
        !pool.try_submit([]() {}).has_value(),
        "try_submit after stop returns nullopt"
    );


    bool rejected = false;

    try
    {
        pool.submit([]() {});
    }
    catch(const std::runtime_error&)
    {
        rejected = true;
    }


    expect(
        rejected,
        "submit after stop throws"
    );


    release.set_value();

    blocker.wait();


    const PoolMetrics m = pool.get_metrics();


    expect(
        m.queue_size == 0,
        "stop(IMMEDIATE) left the queue empty"
    );

    expect(
        m.cancelled == 8,
        "stop(IMMEDIATE) cancelled the queued tasks"
    );

    expect(
        m.running == 0,
        "no task is running after stop"
    );
}


// ------------------------------------------------------------
// clear() and CALLER_RUNS
// ------------------------------------------------------------

static void test_clear_and_caller_runs()
{
    ThreadPool::Config cfg = make_config();

    cfg.min_threads = 1;
    cfg.max_threads = 1;


    ThreadPool pool(cfg);


    std::promise<void> release;

    std::shared_future<void> gate =
        release.get_future().share();


    std::promise<void> started;

    auto blocker =
        pool.submit(
            [gate, &started]()
            {
                started.set_value();

                gate.wait();
            }
        );


    started.get_future().wait();


    std::vector<
        std::future<std::thread::id>
    > ids;


    for(int i = 0; i < 32; ++i)
    {
        ids.push_back(
            pool.submit(
                []()
                {
                    return std::this_thread::get_id();
                }
            )
        );
    }


    const std::size_t dropped = pool.clear();


    expect(
        dropped == 32,
        "clear() reports the number of dropped tasks"
    );

    expect(
        pool.get_metrics().queue_size == 0,
        "clear() empties the queue"
    );


    for(auto& f : ids)
    {
        bool broken = false;

        try
        {
            f.get();
        }
        catch(const std::future_error&)
        {
            broken = true;
        }


        if(!broken)
        {
            expect(
                false,
                "cleared tasks must not stay pending"
            );

            break;
        }
    }


    release.set_value();

    blocker.wait();


    /* CALLER_RUNS: a full queue must not block the producer. */

    ThreadPool::Config runs_cfg = make_config();

    runs_cfg.min_threads = 1;
    runs_cfg.max_threads = 1;
    runs_cfg.max_queue = 1;
    runs_cfg.reject = RejectPolicy::CALLER_RUNS;


    ThreadPool runs(runs_cfg);


    std::promise<void> release2;

    std::shared_future<void> gate2 =
        release2.get_future().share();


    std::promise<void> started2;

    auto blocker2 =
        runs.submit(
            [gate2, &started2]()
            {
                started2.set_value();

                gate2.wait();
            }
        );


    started2.get_future().wait();


    std::atomic<int> caller_ran{0};


    std::vector<std::future<int>> overflow;


    for(int i = 0; i < 4; ++i)
    {
        overflow.push_back(
            runs.submit(
                [&caller_ran]()
                {
                    ++caller_ran;

                    return 0;
                }
            )
        );
    }


    expect(
        caller_ran.load() > 0,
        "CALLER_RUNS executes overflow in the caller"
    );


    release2.set_value();

    blocker2.wait();


    for(auto& f : overflow)
        f.get();
}


// ------------------------------------------------------------
// Runtime resize
// ------------------------------------------------------------

static void test_resize()
{
    ThreadPool::Config cfg = make_config();

    cfg.min_threads = 2;
    cfg.max_threads = 8;
    cfg.idle_timeout =
        std::chrono::milliseconds(300);


    ThreadPool pool(cfg);


    /*
     * Raising the floor must create the missing workers immediately.
     *
     * worker_count_ is incremented inside create_worker_locked() before the
     * thread actually starts, so the metric reflects the new floor without
     * waiting for the threads to spin up.
     */
    pool.resize(5, 8);

    expect(
        pool.get_metrics().worker_count == 5,
        "resize raises worker_count to the new min"
    );


    /*
     * Lower the ceiling and the floor.
     *
     * The 5 idle workers must not be killed mid-task (they have none), but
     * they should shed down to the new min as they go idle.
     */
    pool.resize(2, 4);

    pool.wait_for(
        std::chrono::seconds(5)
    );


    const bool lowered = spin_until(
        [&pool]()
        {
            return pool.get_metrics().worker_count <= 4;
        },
        3000
    );


    expect(
        lowered,
        "resize lowers the burst ceiling"
    );


    const bool settled = spin_until(
        [&pool]()
        {
            return pool.get_metrics().worker_count == 2;
        },
        3000
    );


    expect(
        settled,
        "pool settles at the new min after resize"
    );


    /* Invalid configurations are rejected, same as the constructor. */

    bool threw = false;

    try
    {
        pool.resize(0, 4);
    }
    catch(const std::invalid_argument&)
    {
        threw = true;
    }

    expect(
        threw,
        "resize rejects min_threads == 0"
    );


    threw = false;

    try
    {
        pool.resize(5, 3);
    }
    catch(const std::invalid_argument&)
    {
        threw = true;
    }

    expect(
        threw,
        "resize rejects max_threads < min_threads"
    );
}


// ------------------------------------------------------------
// Runtime configuration
// ------------------------------------------------------------

static void test_runtime_config()
{
    ThreadPool pool(make_config());


    /* A non-positive idle timeout is rejected. */
    bool threw = false;

    try
    {
        pool.set_idle_timeout(
            std::chrono::milliseconds(0)
        );
    }
    catch(const std::invalid_argument&)
    {
        threw = true;
    }

    expect(
        threw,
        "set_idle_timeout rejects <= 0"
    );


    /* A valid idle timeout does not disturb normal execution. */
    pool.set_idle_timeout(
        std::chrono::milliseconds(150)
    );

    auto v = pool.submit(
        []()
        {
            return 123;
        }
    );

    expect(
        v.get() == 123,
        "pool still works after set_idle_timeout"
    );


    /*
     * set_reject_policy must take effect on the next full-queue submit.
     *
     * With a single worker and a full queue, BLOCK would make submit()
     * block forever; switching to DISCARD must make it throw instead.
     */
    ThreadPool::Config cfg = make_config();

    cfg.min_threads = 1;
    cfg.max_threads = 1;
    cfg.max_queue = 1;

    ThreadPool limited(cfg);


    std::promise<void> release;

    std::shared_future<void> gate =
        release.get_future().share();

    std::promise<void> started;

    auto blocker =
        limited.submit(
            [gate, &started]()
            {
                started.set_value();

                gate.wait();
            }
        );

    started.get_future().wait();


    /*
     * The worker is stuck on the blocker, so this task lands in the queue
     * and fills it (max_queue == 1).
     */
    auto queued =
        limited.submit(
            []()
            {
                return 0;
            }
        );


    /* Now the queue is full. Switch the policy, then submit again. */
    limited.set_reject_policy(
        RejectPolicy::DISCARD
    );


    bool rejected = false;

    try
    {
        limited.submit(
            []()
            {
                return 0;
            }
        );
    }
    catch(const std::runtime_error&)
    {
        rejected = true;
    }

    expect(
        rejected,
        "set_reject_policy(DISCARD) makes a full queue throw"
    );


    (void)queued;


    release.set_value();

    blocker.wait();
}


// ------------------------------------------------------------
// Scheduled tasks, parallel algorithms, local (nested) queue
// ------------------------------------------------------------

static void test_scheduled_and_parallel()
{
    using namespace std::chrono_literals;


    /* submit_after() runs only after its delay. */
    {
        ThreadPool pool(make_config());

        auto f = pool.submit_after(
            150ms,
            []()
            {
                return 7;
            }
        );

        /*
         * Before the delay elapses the result must not be ready.
         */
        const bool not_ready =
            f.wait_for(20ms)
                == std::future_status::timeout;

        expect(
            not_ready,
            "submit_after is not ready before its delay"
        );

        expect(
            f.get() == 7,
            "submit_after runs after the delay"
        );
    }


    /*
     * Cancelling a deferred task before its deadline breaks its future
     * with broken_promise instead of running it.
     */
    {
        ThreadPool pool(make_config());

        auto h = pool.submit_cancellable_after(
            300ms,
            0,
            []()
            {
                return 1;
            }
        );

        const bool cancelled = h.handle.cancel();

        expect(
            cancelled,
            "scheduled task can be cancelled before it runs"
        );

        bool threw = false;

        try
        {
            h.future.get();
        }
        catch(const std::future_error&)
        {
            threw = true;
        }

        expect(
            threw,
            "cancelling a scheduled task breaks its future"
        );
    }


    /* parallel_for covers every index exactly once. */
    {
        ThreadPool pool(make_config());

        const std::size_t n = 100000;

        std::atomic<std::size_t> count{0};

        pool.parallel_for(
            0,
            n,
            1024,
            [&count](std::size_t i)
            {
                (void)i;

                count.fetch_add(
                    1,
                    std::memory_order_relaxed
                );
            }
        ).get();

        expect(
            count.load() == n,
            "parallel_for covers the whole range"
        );
    }


    /* parallel_reduce computes the correct total. */
    {
        ThreadPool pool(make_config());

        const std::size_t n = 100000;

        const std::size_t sum =
            pool.parallel_reduce(
                0,
                n,
                1024,
                std::size_t(0),
                [](std::size_t b, std::size_t e)
                {
                    std::size_t s = 0;

                    for(std::size_t i = b; i < e; ++i)
                        s += i;

                    return s;
                },
                [](std::size_t a, std::size_t b)
                {
                    return a + b;
                }
            ).get();

        const std::size_t expected =
            n * (n - 1) / 2;

        expect(
            sum == expected,
            "parallel_reduce computes the correct sum"
        );
    }


    /*
     * A task that submits more work from inside the pool routes those
     * nested tasks onto the worker's own local queue.
     *
     * wait() counts local_pending_, so it drains them too.
     */
    {
        ThreadPool pool(make_config());

        std::atomic<int> nested{0};


        pool.submit(
            [&pool, &nested]()
            {
                for(int i = 0; i < 50; ++i)
                {
                    pool.submit(
                        [&nested]()
                        {
                            nested.fetch_add(
                                1,
                                std::memory_order_relaxed
                            );
                        }
                    );
                }
            }
        ).get();

        pool.wait();

        expect(
            nested.load() == 50,
            "nested submits run via the worker local queue"
        );
    }
}


// ------------------------------------------------------------
// Runtime batch_size tuning + lightweight accessors
// ------------------------------------------------------------

static void test_runtime_batch_and_accessors()
{
    using namespace std::chrono_literals;

    /* queue_size / worker_count / active_count accounting. */
    {
        ThreadPool::Config cfg = make_config();
        cfg.min_threads = 1;
        cfg.max_threads = 1;
        cfg.max_queue = 1024;
        ThreadPool pool(cfg);

        std::promise<void> release;
        std::future<void> blocker = release.get_future();
        std::promise<void> started;
        std::future<void> started_f = started.get_future();

        pool.submit(
            [&blocker, &started]()
            {
                started.set_value();
                blocker.wait();
            }
        );

        /* Ensure the worker is genuinely inside blocker.wait() before we
         * queue anything, otherwise the blocker task itself could still be
         * sitting in the global queue and skew queue_size(). */
        started_f.wait();

        /* Queue a handful of runnable tasks behind the blocker. */
        for(int i = 0; i < 5; ++i)
            pool.submit([](){});

        expect(
            pool.queue_size() == 5,
            "queue_size reflects the waiting runnable tasks"
        );

        expect(
            pool.worker_count() == 1,
            "worker_count matches the single worker"
        );

        expect(
            pool.active_count() >= 6,
            "active_count counts the running plus queued tasks"
        );

        release.set_value();
        blocker.wait();
        pool.wait();

        expect(
            pool.queue_size() == 0,
            "queue_size drains to zero after wait"
        );
    }

    /* scheduled_count tracks not-yet-due scheduled tasks. */
    {
        ThreadPool pool(make_config());

        pool.submit_after(2s, [](){ return 1; });
        pool.submit_after(2s, [](){ return 2; });

        expect(
            pool.scheduled_count() == 2,
            "scheduled_count reflects enqueued but not-yet-due tasks"
        );

        /* The futures are intentionally left unobserved; the pool drops
         * not-yet-due scheduled tasks on shutdown, so teardown is clean. */
    }

    /* set_batch_size updates the config and clamps values < 1 to 1. */
    {
        ThreadPool pool(make_config());

        pool.set_batch_size(16);
        expect(
            pool.get_config().batch_size == 16,
            "set_batch_size stores the new value"
        );

        pool.set_batch_size(0);
        expect(
            pool.get_config().batch_size == 1,
            "set_batch_size clamps values below 1 to 1"
        );
    }
}


// ------------------------------------------------------------
// Concurrency stress: lock-free completion accounting must not
// lose a wakeup (wait() must not deadlock)
// ------------------------------------------------------------

static void test_wait_stress()
{
    using namespace std::chrono_literals;

    ThreadPool pool(make_config());

    std::atomic<std::uint64_t> done{0};

    /*
     * Several external producers hammer submit() + get() while a separate
     * thread calls wait(). This exercises the path where inflight_ ticks
     * down to zero lock-free; a lost wakeup would hang the waiter.
     */
    const int producer_count = 4;
    const std::uint64_t per = 20000;

    std::vector<std::thread> producers;
    producers.reserve(producer_count);

    for(int p = 0; p < producer_count; ++p)
    {
        producers.emplace_back(
            [&pool, &done, per]()
            {
                for(std::uint64_t i = 0; i < per; ++i)
                {
                    pool.submit(
                        [&done]()
                        {
                            done.fetch_add(
                                1,
                                std::memory_order_relaxed
                            );
                        }
                    ).get();
                }
            }
        );
    }

    std::thread waiter(
        [&pool]()
        {
            pool.wait();
        }
    );

    for(auto& t : producers)
        t.join();

    waiter.join();

    /* Authoritative re-check once everything has been submitted. */
    pool.wait();

    expect(
        done.load()
            == static_cast<std::uint64_t>(producer_count) * per,
        "all stress tasks executed without a lost wakeup"
    );
}


static void test_scaling_churn()
{
    using namespace std::chrono_literals;

    ThreadPool::Config cfg = make_config();

    cfg.min_threads = 2;
    cfg.max_threads = 16;
    cfg.max_queue = 8192;
    cfg.idle_timeout = 120ms;

    ThreadPool pool(cfg);

    std::atomic<std::uint64_t> done{0};


    /*
     * Many rounds of "burst -> drain -> idle" drive the pool through a
     * large number of worker create / reap cycles. Every create now spawns
     * its thread outside of the pool lock and every reap joins outside of
     * it, so this is a direct stress test of the 3.1.1 lock refactor.
     */
    const int rounds = 40;
    const int burst = 300;

    for(int r = 0; r < rounds; ++r)
    {
        std::vector<std::future<void>> futs;
        futs.reserve(burst);

        for(int i = 0; i < burst; ++i)
        {
            futs.push_back(
                pool.submit(
                    [&done]()
                    {
                        /*
                         * A little real work so workers stay busy long
                         * enough for the pool to expand under the burst.
                         */
                        volatile std::uint64_t acc = 0;

                        for(int k = 0; k < 200; ++k)
                            acc += k;

                        done.fetch_add(
                            1,
                            std::memory_order_relaxed
                        );

                        (void)acc;
                    }
                )
            );
        }

        for(auto& f : futs)
            f.get();

        /* Let the pool idle out and reclaim the extra workers. */
        sleep_ms(220);
    }

    const std::uint64_t expected =
        static_cast<std::uint64_t>(rounds) * burst;

    expect(
        done.load() == expected,
        "every churn-round task completed"
    );


    /*
     * After idling, the pool must have settled back to the minimum and the
     * reaper must not have leaked any slots.
     */
    sleep_ms(300);

    const PoolMetrics m = pool.get_metrics();

    expect(
        m.worker_count <= cfg.max_threads,
        "worker count never exceeds max_threads"
    );

    expect(
        m.slots <= cfg.max_threads,
        "live slots never exceed max_threads"
    );


    /*
     * The pool is still usable after all that churn.
     */
    auto probe =
        pool.submit(
            []()
            {
                return 7;
            }
        );

    expect(
        probe.get() == 7,
        "pool still executes tasks after heavy scaling churn"
    );
}


// ------------------------------------------------------------
// 4.0.0: work-stealing scheduler + future composition + default pool
// ------------------------------------------------------------

static void test_work_stealing_and_composition()
{
    using namespace std::chrono_literals;

    /*
     * Work-stealing: a fork-join backlog created on one worker must be
     * drained by an idle peer rather than stalling on the producer.
     */
    {
        ThreadPool::Config cfg = make_config();

        cfg.min_threads = 2;
        cfg.max_threads = 2;
        cfg.idle_timeout = 500ms;

        ThreadPool pool(cfg);

        std::atomic<int> nested{0};

        pool.submit(
            [&pool, &nested]()
            {
                for(int i = 0; i < 500; ++i)
                {
                    pool.submit(
                        [&nested]()
                        {
                            sleep_ms(2);

                            nested.fetch_add(
                                1,
                                std::memory_order_relaxed
                            );
                        }
                    );
                }
            }
        ).get();

        pool.wait();

        expect(
            nested.load() == 500,
            "nested fork-join tasks all complete"
        );

        expect(
            pool.get_metrics().stolen > 0,
            "idle worker stole the fork-join backlog (stealing active)"
        );
    }

    /* when_all collects every result in submission order. */
    {
        ThreadPool pool(make_config());

        std::vector<std::future<int>> futs;

        for(int i = 1; i <= 4; ++i)
            futs.push_back(
                pool.submit([i]()
                {
                    return i * i;
                })
            );

        auto all = pool.when_all(
            std::move(futs)
        ).get();

        std::vector<int> expected{1, 4, 9, 16};

        expect(
            all == expected,
            "when_all returns results in submission order"
        );
    }

    /* when_all(void) waits for every task to finish. */
    {
        ThreadPool pool(make_config());

        std::atomic<int> count{0};

        std::vector<std::future<void>> futs;

        for(int i = 0; i < 16; ++i)
            futs.push_back(
                pool.submit(
                    [&count]()
                    {
                        count.fetch_add(
                            1,
                            std::memory_order_relaxed
                        );
                    }
                )
            );

        pool.when_all(
            std::move(futs)
        ).get();

        expect(
            count.load() == 16,
            "when_all(void) waits for every task"
        );
    }

    /*
     * when_any returns as soon as the first future is ready, without
     * blocking on the delayed ones.
     */
    {
        ThreadPool pool(make_config());

        std::vector<std::future<int>> futs;

        futs.push_back(
            pool.submit(
                []()
                {
                    return 42;
                }
            )
        );

        futs.push_back(
            pool.submit_after(
                250ms,
                []()
                {
                    return 1;
                }
            )
        );

        futs.push_back(
            pool.submit_after(
                250ms,
                []()
                {
                    return 2;
                }
            )
        );

        auto start =
            std::chrono::steady_clock::now();

        auto res = pool.when_any(
            std::move(futs)
        ).get();

        auto elapsed =
            std::chrono::steady_clock::now()
                - start;

        expect(
            res.second == 42,
            "when_any returns the first-ready result"
        );

        expect(
            elapsed < 100ms,
            "when_any does not wait for the delayed futures"
        );
    }

    /* when_any(void) reports the index of the first-ready task. */
    {
        ThreadPool pool(make_config());

        std::vector<std::future<void>> futs;

        futs.push_back(
            pool.submit(
                [](){}
            )
        );

        futs.push_back(
            pool.submit_after(
                250ms,
                [](){}
            )
        );

        auto start =
            std::chrono::steady_clock::now();

        auto idx = pool.when_any(
            std::move(futs)
        ).get();

        auto elapsed =
            std::chrono::steady_clock::now()
                - start;

        expect(
            idx == 0,
            "when_any(void) reports the first-ready index"
        );

        expect(
            elapsed < 100ms,
            "when_any(void) does not wait for the delayed task"
        );
    }

    /* default_pool is a usable process-wide singleton. */
    {
        auto& a = ThreadPool::default_pool();
        auto& b = ThreadPool::default_pool();

        expect(
            &a == &b,
            "default_pool returns the same singleton"
        );

        auto v = a.submit(
            []()
            {
                return 7;
            }
        );

        expect(
            v.get() == 7,
            "default_pool executes a submitted task"
        );
    }
}


/* ------------------------------------------------------------------
 * 5.0.0: task context, cooperative cancellation, continuations,
 * batch submission and structured concurrency.
 * ------------------------------------------------------------------ */

void test_v5_context_and_cancellation()
{
    ThreadPool pool;


    /* Outside a task there is no context at all. */
    expect(
        ThreadPool::current_worker_id()
            == ThreadPool::no_worker,
        "context: no worker id outside a task"
    );

    expect(
        ThreadPool::current_pool() == nullptr,
        "context: no owning pool outside a task"
    );

    expect(
        !ThreadPool::current_token().cancellable(),
        "context: no token outside a task"
    );


    /* A classic task still runs, but has no attached token. */
    auto plain = pool.submit(
        [&pool]()
        {
            return ThreadPool::current_worker_id()
                    != ThreadPool::no_worker
                &&
                ThreadPool::current_pool()
                    == &pool
                &&
                !ThreadPool::current_token()
                    .cancellable();
        }
    );

    expect(
        plain.get(),
        "context: worker id / pool / empty token in task"
    );


    /* A token task sees exactly its own cancellation domain. */
    CancellationSource src;

    auto attached = pool.submit_token(
        src.token(),
        [&src]()
        {
            return ThreadPool::current_token()
                    .cancellable()
                &&
                !ThreadPool::current_token()
                    .is_cancelled()
                &&
                src.token().is_cancelled()
                    == ThreadPool::current_token()
                        .is_cancelled();
        }
    );

    expect(
        attached.get(),
        "context: submit_token installs its token"
    );


    /* Cancelling before submission skips the callable. */
    CancellationSource dead;

    expect(
        dead.request_cancel()
            &&
        !dead.request_cancel(),
        "CancellationSource::request_cancel is idempotent"
    );

    auto skipped = pool.submit_token(
        dead.token(),
        []()
        {
            return 11;
        }
    );

    bool got_task_cancelled = false;

    try
    {
        (void)skipped.get();
    }
    catch(const TaskCancelled&)
    {
        got_task_cancelled = true;
    }
    catch(...)
    {
    }

    expect(
        got_task_cancelled,
        "token: cancelled task future carries TaskCancelled"
    );

    expect(
        pool.get_metrics().token_cancelled >= 1,
        "metrics: token_cancelled is counted"
    );


    /* A running task observes cancellation through current_token(). */
    CancellationSource live;

    std::promise<void> started;
    auto started_future = started.get_future();

    std::atomic<bool> observed{false};

    auto cooperating = pool.submit_token(
        live.token(),
        [&started, &observed]()
        {
            started.set_value();

            for(int i = 0; i < 400; ++i)
            {
                if(ThreadPool::current_token()
                    .is_cancelled())
                {
                    observed.store(true);

                    ThreadPool::current_token()
                        .throw_if_cancelled();
                }

                sleep_ms(1);
            }
        }
    );

    started_future.wait();

    live.request_cancel();

    bool cooperative_cancel = false;

    try
    {
        cooperating.get();
    }
    catch(const TaskCancelled&)
    {
        cooperative_cancel = true;
    }
    catch(...)
    {
    }

    expect(
        cooperative_cancel && observed.load(),
        "token: running task cooperatively cancels"
    );


    /* The other token-aware entry points stay usable. */
    auto handle_result = pool.submit_cancellable_token(
        3,
        CancellationToken{},
        []()
        {
            return 4;
        }
    );

    expect(
        handle_result.handle.valid()
            &&
        handle_result.future.get() == 4,
        "submit_cancellable_token returns handle + value"
    );

    auto optional_result = pool.try_submit_token(
        CancellationToken{},
        []()
        {
            return 5;
        }
    );

    expect(
        optional_result.has_value()
            &&
        optional_result->get() == 5,
        "try_submit_token accepts a task"
    );

    auto deferred = pool.submit_after_token(
        std::chrono::milliseconds(5),
        dead.token(),
        []()
        {
            return 6;
        }
    );

    bool deferred_cancelled = false;

    try
    {
        (void)deferred.get();
    }
    catch(const TaskCancelled&)
    {
        deferred_cancelled = true;
    }
    catch(...)
    {
    }

    expect(
        deferred_cancelled,
        "submit_after_token honours a cancelled token"
    );


    /* User exceptions are still delivered, and now counted. */
    const std::uint64_t failed_before =
        pool.get_metrics().failed;

    auto throwing = pool.submit(
        []() -> int
        {
            throw std::runtime_error("boom");
        }
    );

    bool user_exception = false;

    try
    {
        (void)throwing.get();
    }
    catch(const std::runtime_error&)
    {
        user_exception = true;
    }

    expect(
        user_exception
            &&
        pool.get_metrics().failed
            > failed_before,
        "metrics: failed counts user exceptions"
    );


    pool.wait();
}


void test_v5_continuations_batches_and_groups()
{
    ThreadPool pool;


    /* then() on a value future. */
    auto chained = pool.then(
        pool.submit(
            []()
            {
                return 3;
            }
        ),
        [](int x)
        {
            return x * 2;
        }
    );

    expect(
        chained.get() == 6,
        "then(value) chains without blocking"
    );


    /* then() on a void future, with a value and with a void result. */
    auto from_void = pool.then(
        pool.submit(
            []()
            {
            }
        ),
        []()
        {
            return 9;
        }
    );

    std::atomic<int> side_effect{0};

    auto to_void = pool.then(
        pool.submit(
            []()
            {
            }
        ),
        [&side_effect]()
        {
            ++side_effect;
        }
    );

    to_void.get();

    expect(
        from_void.get() == 9
            &&
        side_effect.load() == 1,
        "then(void) supports value and void results"
    );


    /* An exception in the predecessor skips the continuation. */
    auto bad = pool.submit(
        []() -> int
        {
            throw std::runtime_error("boom");
        }
    );

    auto propagated = pool.then(
        std::move(bad),
        [](int x)
        {
            return x + 1;
        }
    );

    bool exception_propagated = false;

    try
    {
        (void)propagated.get();
    }
    catch(const std::runtime_error&)
    {
        exception_propagated = true;
    }

    expect(
        exception_propagated,
        "then() propagates predecessor exceptions"
    );


    /* Batch submission keeps index order. */
    auto batch = pool.submit_batch(
        8,
        [](std::size_t i)
        {
            return i * i;
        }
    );

    bool batch_ok = (batch.size() == 8);

    for(std::size_t i = 0; i < batch.size(); ++i)
    {
        if(batch[i].get() != i * i)
            batch_ok = false;
    }

    expect(
        batch_ok,
        "submit_batch preserves index order"
    );

    auto priority_batch = pool.submit_batch_priority(
        7,
        3,
        [](std::size_t i)
        {
            return static_cast<int>(i) + 1;
        }
    );

    expect(
        priority_batch.size() == 3
            &&
        priority_batch[0].get() == 1
            &&
        priority_batch[2].get() == 3,
        "submit_batch_priority works"
    );


    /* A group waits only for its own tasks. */
    std::promise<void> outside_started;
    auto outside_started_future =
        outside_started.get_future();

    auto outside = pool.submit(
        [&outside_started]()
        {
            outside_started.set_value();

            sleep_ms(200);
        }
    );

    outside_started_future.wait();


    TaskGroup group(pool);

    auto group_value = group.submit(
        []()
        {
            return 42;
        }
    );

    expect(
        group.wait_for(std::chrono::milliseconds(500))
            &&
        group_value.get() == 42,
        "TaskGroup::wait_for waits only for the group"
    );

    expect(
        group.pending() == 0
            &&
        group.submitted() == 1
            &&
        group.failed() == 0,
        "TaskGroup counters are accurate"
    );

    outside.get();


    /* Group cancellation is cooperative and never leaks pending work. */
    ThreadPool::Config single_cfg;
    single_cfg.min_threads = 1;
    single_cfg.max_threads = 1;

    ThreadPool single(single_cfg);

    {
        TaskGroup cancelling(single);

        std::promise<void> blocker_started;
        auto blocker_future =
            blocker_started.get_future();

        auto blocker = cancelling.submit(
            [&blocker_started]()
            {
                blocker_started.set_value();

                sleep_ms(120);

                return 1;
            }
        );

        blocker_future.wait();


        std::vector<std::future<int>> queued;

        for(int i = 0; i < 6; ++i)
        {
            queued.push_back(
                cancelling.submit(
                    [i]()
                    {
                        return i;
                    }
                )
            );
        }


        expect(
            cancelling.request_cancel()
                &&
            cancelling.is_cancelled(),
            "TaskGroup::request_cancel flips the token"
        );


        cancelling.wait();

        int cancelled_count = 0;

        for(auto& f : queued)
        {
            try
            {
                (void)f.get();
            }
            catch(const TaskCancelled&)
            {
                ++cancelled_count;
            }
            catch(...)
            {
            }
        }

        expect(
            cancelled_count >= 1,
            "TaskGroup: queued tasks observe cancellation"
        );

        expect(
            cancelling.pending() == 0
                &&
            blocker.get() == 1,
            "TaskGroup: pending reaches zero after cancel"
        );


        bool submit_after_cancel = false;

        try
        {
            (void)cancelling.submit(
                []()
                {
                    return 0;
                }
            );
        }
        catch(const TaskCancelled&)
        {
            submit_after_cancel = true;
        }
        catch(...)
        {
        }

        expect(
            submit_after_cancel,
            "TaskGroup rejects submissions after cancel"
        );
    }


    /* Exception aggregation. */
    {
        TaskGroup failing(pool);

        std::vector<std::future<int>> futures;

        for(int i = 0; i < 3; ++i)
        {
            futures.push_back(
                failing.submit(
                    []() -> int
                    {
                        throw std::runtime_error("bad");
                    }
                )
            );
        }

        failing.wait();

        bool aggregated = false;

        try
        {
            failing.rethrow_exceptions();
        }
        catch(const TaskGroupError& e)
        {
            aggregated =
                e.size() == 3
                &&
                e.errors().size() == 3;
        }

        expect(
            failing.failed() == 3
                &&
            aggregated,
            "TaskGroup aggregates task exceptions"
        );
    }


    /* RAII: the destructor waits, CANCEL_AND_WAIT cancels first. */
    std::atomic<int> finished{0};

    {
        TaskGroup raii(pool);

        for(int i = 0; i < 4; ++i)
        {
            (void)raii.submit(
                [&finished]()
                {
                    sleep_ms(10);

                    ++finished;
                }
            );
        }
    }

    expect(
        finished.load() == 4,
        "TaskGroup destructor waits for all tasks"
    );


    CancellationToken observed_token;

    std::atomic<int> cooperative_done{0};

    {
        TaskGroup auto_cancel(
            pool,
            GroupDestroyPolicy::CANCEL_AND_WAIT
        );

        observed_token = auto_cancel.token();

        std::promise<void> started;
        auto started_future = started.get_future();

        (void)auto_cancel.submit(
            [&started, &cooperative_done, &observed_token]()
            {
                started.set_value();

                while(!observed_token.is_cancelled())
                    sleep_ms(1);

                ++cooperative_done;
            }
        );

        started_future.wait();
    }

    expect(
        observed_token.is_cancelled()
            &&
        cooperative_done.load() == 1,
        "CANCEL_AND_WAIT cancels then joins"
    );


    /* wait() from a worker must be diagnosed, not deadlocked. */
    auto from_worker = pool.submit(
        [&pool]()
        {
            TaskGroup nested(pool);

            (void)nested.submit(
                []()
                {
                    return 1;
                }
            );

            try
            {
                nested.wait();

                return false;
            }
            catch(const std::logic_error&)
            {
                return true;
            }
        }
    );

    expect(
        from_worker.get(),
        "TaskGroup::wait rejects worker-thread calls"
    );


    pool.wait();
}


/* ------------------------------------------------------------------
 * 5.0.1 regressions: a discarded task must never strand a TaskGroup.
 * ------------------------------------------------------------------ */

void test_v501_discard_and_context()
{
    ThreadPool::Config cfg;
    cfg.min_threads = 1;
    cfg.max_threads = 1;


    /* clear() drops queued group tasks before their wrappers can run. */
    {
        ThreadPool pool(cfg);

        auto group =
            std::make_unique<TaskGroup>(pool);

        std::promise<void> started;
        auto started_future = started.get_future();

        auto blocker = group->submit(
            [&started]()
            {
                started.set_value();

                sleep_ms(120);

                return 1;
            }
        );

        started_future.wait();


        std::vector<std::future<int>> queued;

        for(int i = 0; i < 5; ++i)
        {
            queued.push_back(
                group->submit(
                    [i]()
                    {
                        return i;
                    }
                )
            );
        }


        const std::size_t dropped = pool.clear();

        expect(
            dropped >= 5,
            "clear() discards the queued group tasks"
        );


        const bool drained =
            group->wait_for(
                std::chrono::milliseconds(1000)
            );

        expect(
            drained
                &&
            group->pending() == 0,
            "clear() does not strand TaskGroup::wait()"
        );


        if(drained)
        {
            int cancelled_count = 0;

            for(auto& f : queued)
            {
                try
                {
                    (void)f.get();
                }
                catch(const TaskCancelled&)
                {
                    ++cancelled_count;
                }
                catch(...)
                {
                }
            }

            expect(
                cancelled_count >= 1,
                "discarded group futures carry TaskCancelled"
            );

            expect(
                blocker.get() == 1,
                "clear() leaves the running task alone"
            );
        }
        else
        {
            expect(
                false,
                "discarded group futures carry TaskCancelled"
            );

            expect(
                false,
                "clear() leaves the running task alone"
            );

            /* Deliberate leak: never hang the test binary on a regression. */
            group.release();
        }
    }


    /* stop(IMMEDIATE) takes the same discard path. */
    {
        ThreadPool pool(cfg);

        auto group =
            std::make_unique<TaskGroup>(pool);

        std::promise<void> started;
        auto started_future = started.get_future();

        auto blocker = group->submit(
            [&started]()
            {
                started.set_value();

                sleep_ms(120);

                return 1;
            }
        );

        started_future.wait();


        std::vector<std::future<int>> queued;

        for(int i = 0; i < 5; ++i)
        {
            queued.push_back(
                group->submit(
                    [i]()
                    {
                        return i;
                    }
                )
            );
        }


        pool.stop(PoolStopMode::IMMEDIATE);

        const bool drained =
            group->wait_for(
                std::chrono::milliseconds(1000)
            );

        expect(
            drained
                &&
            group->pending() == 0,
            "stop(IMMEDIATE) does not strand TaskGroup::wait()"
        );


        if(drained)
        {
            int cancelled_count = 0;

            for(auto& f : queued)
            {
                try
                {
                    (void)f.get();
                }
                catch(const TaskCancelled&)
                {
                    ++cancelled_count;
                }
                catch(...)
                {
                }
            }

            expect(
                cancelled_count >= 1,
                "stop(IMMEDIATE) fulfils discarded futures"
            );

            expect(
                blocker.get() == 1,
                "stop(IMMEDIATE) leaves the running task alone"
            );
        }
        else
        {
            expect(
                false,
                "stop(IMMEDIATE) fulfils discarded futures"
            );

            expect(
                false,
                "stop(IMMEDIATE) leaves the running task alone"
            );

            group.release();
        }
    }


    /* A group task now sees the group token in the execution context. */
    {
        ThreadPool pool(cfg);

        TaskGroup group(pool);

        auto token = group.token();

        auto value = group.submit(
            [&token]()
            {
                return ThreadPool::current_token()
                        .cancellable()
                    &&
                ThreadPool::current_token()
                        .is_cancelled()
                    == token.is_cancelled();
            }
        );

        expect(
            value.get(),
            "group task exposes its token via current_token()"
        );

        group.wait();
    }


    /* Cooperative cancellation through current_token() inside a group. */
    {
        ThreadPool pool(cfg);

        auto group =
            std::make_unique<TaskGroup>(
                pool,
                GroupDestroyPolicy::CANCEL_AND_WAIT
            );

        std::promise<void> started;
        auto started_future = started.get_future();

        std::atomic<bool> observed{false};

        auto task = group->submit(
            [&started, &observed]()
            {
                started.set_value();

                for(int i = 0; i < 400; ++i)
                {
                    if(ThreadPool::current_token()
                        .is_cancelled())
                    {
                        observed.store(true);

                        ThreadPool::current_token()
                            .throw_if_cancelled();
                    }

                    sleep_ms(1);
                }
            }
        );

        started_future.wait();

        group->request_cancel();

        bool threw = false;

        try
        {
            task.get();
        }
        catch(const TaskCancelled&)
        {
            threw = true;
        }
        catch(...)
        {
        }

        expect(
            threw && observed.load(),
            "group task cancels via current_token()"
        );

        const bool drained =
            group->wait_for(
                std::chrono::milliseconds(1000)
            );

        expect(
            drained,
            "cooperative group cancel drains pending"
        );

        if(!drained)
            group.release();
    }


}


void test_v510_deadlines()
{
    using clock = std::chrono::steady_clock;


    /* A default token has neither a flag nor a deadline. */
    {
        CancellationToken token;

        expect(
            !token.cancellable()
                &&
            !token.is_cancelled()
                &&
            !token.timed_out()
                &&
            token.deadline()
                == clock::time_point{},
            "v5.1 deadline: default token stays inert"
        );
    }


    /* Deadlines are evaluated lazily: no timer thread is involved. */
    {
        CancellationSource source;

        source.cancel_after(
            std::chrono::milliseconds(40)
        );

        CancellationToken token = source.token();

        expect(
            !source.is_cancelled()
                &&
            !token.is_cancelled()
                &&
            token.deadline() > clock::now(),
            "v5.1 deadline: not cancelled before expiry"
        );


        sleep_ms(90);

        expect(
            source.is_cancelled()
                &&
            token.is_cancelled()
                &&
            source.timed_out()
                &&
            token.timed_out(),
            "v5.1 deadline: expires lazily on check"
        );


        bool message_ok = false;

        try
        {
            token.throw_if_cancelled();
        }
        catch(const TaskCancelled& e)
        {
            message_ok =
                std::string(e.what()).find(
                    "timed out"
                ) != std::string::npos;
        }
        catch(...)
        {
        }

        expect(
            message_ok,
            "v5.1 deadline: TaskCancelled reports timeout"
        );
    }


    /* An explicit request_cancel() is not reported as a timeout. */
    {
        CancellationSource source;

        source.request_cancel();

        bool message_ok = false;

        try
        {
            source.token().throw_if_cancelled();
        }
        catch(const TaskCancelled& e)
        {
            message_ok =
                !source.timed_out()
                    &&
                std::string(e.what()).find(
                    "timed out"
                ) == std::string::npos;
        }
        catch(...)
        {
        }

        expect(
            message_ok,
            "v5.1 deadline: explicit cancel is not a timeout"
        );
    }


    /* Repeated cancel_at() calls keep the earliest deadline. */
    {
        CancellationSource source;

        const auto now = clock::now();

        source.cancel_at(
            now + std::chrono::seconds(10)
        );

        source.cancel_at(
            now + std::chrono::milliseconds(20)
        );

        CancellationToken token = source.token();

        expect(
            token.deadline()
                < now + std::chrono::seconds(1),
            "v5.1 deadline: earliest cancel_at wins"
        );


        sleep_ms(60);

        expect(
            token.is_cancelled(),
            "v5.1 deadline: earlier deadline expires"
        );
    }


    /* Deadline-bound submissions that still have time simply run. */
    {
        ThreadPool pool;

        auto quick =
            pool.submit_with_timeout(
                std::chrono::seconds(5),
                [](int x)
                {
                    return x * 2;
                },
                21
            );

        auto absolute =
            pool.submit_with_deadline(
                clock::now() + std::chrono::seconds(5),
                []()
                {
                    return 9;
                }
            );

        auto priority =
            pool.submit_with_timeout_priority(
                3,
                std::chrono::seconds(5),
                []()
                {
                    return 11;
                }
            );

        auto priority_at =
            pool.submit_with_deadline_priority(
                3,
                clock::now() + std::chrono::seconds(5),
                []()
                {
                    return 13;
                }
            );


        expect(
            quick.future.get() == 42
                &&
            absolute.future.get() == 9
                &&
            priority.future.get() == 11
                &&
            priority_at.future.get() == 13,
            "v5.1 deadline: unexpired tasks run normally"
        );


        expect(
            quick.handle.valid()
                &&
            absolute.handle.valid(),
            "v5.1 deadline: result still carries a TaskHandle"
        );

        pool.wait();
    }


    /* An already-expired deadline skips the callable entirely. */
    {
        ThreadPool pool;

        auto task =
            pool.submit_with_deadline(
                clock::now()
                    - std::chrono::seconds(1),
                []()
                {
                    return 1;
                }
            );

        bool ok = false;

        try
        {
            (void)task.future.get();
        }
        catch(const TaskCancelled& e)
        {
            ok =
                std::string(e.what()).find(
                    "timed out"
                ) != std::string::npos;
        }
        catch(...)
        {
        }

        expect(
            ok,
            "v5.1 deadline: expired task future carries timeout"
        );


        expect(
            pool.get_metrics().timed_out >= 1
                &&
            pool.get_metrics().token_cancelled >= 1,
            "v5.1 deadline: timed_out metric is recorded"
        );

        pool.wait();
    }


    /* A task queued behind a blocker expires before it can start. */
    {
        ThreadPool::Config cfg;

        cfg.min_threads = 1;
        cfg.max_threads = 1;

        ThreadPool pool(cfg);

        std::promise<void> started;
        std::promise<void> release;

        auto started_future = started.get_future();

        /*
         * shared_future is copyable: MSVC rejects an implicit capture of
         * a move-only std::future<void> inside the submitted callable.
         */
        std::shared_future<void> release_future =
            release.get_future().share();

        auto blocker =
            pool.submit(
                [&started, release_future]()
                {
                    started.set_value();

                    release_future.wait();
                }
            );

        started_future.wait();


        auto queued =
            pool.submit_with_timeout(
                std::chrono::milliseconds(30),
                []()
                {
                    return 7;
                }
            );

        sleep_ms(90);

        release.set_value();

        blocker.get();


        bool ok = false;

        try
        {
            (void)queued.future.get();
        }
        catch(const TaskCancelled&)
        {
            ok = true;
        }
        catch(...)
        {
        }

        expect(
            ok,
            "v5.1 deadline: queued task expires before start"
        );

        pool.wait();
    }


    /* A running task can observe its own deadline through the context. */
    {
        ThreadPool pool;

        auto task =
            pool.submit_with_timeout(
                std::chrono::milliseconds(30),
                []()
                {
                    for(int i = 0; i < 100; ++i)
                    {
                        sleep_ms(3);

                        ThreadPool::current_token()
                            .throw_if_cancelled();
                    }

                    return 1;
                }
            );

        bool ok = false;

        try
        {
            (void)task.future.get();
        }
        catch(const TaskCancelled&)
        {
            ok = true;
        }
        catch(...)
        {
        }

        expect(
            ok,
            "v5.1 deadline: running task sees current_token"
        );

        pool.wait();
    }


    /* Group-wide deadlines cancel queued members without hanging wait(). */
    {
        ThreadPool::Config cfg;

        cfg.min_threads = 1;
        cfg.max_threads = 1;

        ThreadPool pool(cfg);

        std::promise<void> started;
        std::promise<void> release;

        auto started_future = started.get_future();

        /*
         * shared_future is copyable: MSVC rejects an implicit capture of
         * a move-only std::future<void> inside the submitted callable.
         */
        std::shared_future<void> release_future =
            release.get_future().share();

        auto blocker =
            pool.submit(
                [&started, release_future]()
                {
                    started.set_value();

                    release_future.wait();
                }
            );

        started_future.wait();


        auto group =
            std::make_unique<TaskGroup>(pool);

        auto a = group->submit(
            []()
            {
                return 1;
            }
        );

        auto b = group->submit(
            []()
            {
                return 2;
            }
        );

        group->cancel_after(
            std::chrono::milliseconds(20)
        );

        sleep_ms(80);

        release.set_value();

        blocker.get();


        const bool drained =
            group->wait_for(
                std::chrono::seconds(2)
            );

        expect(
            drained
                &&
            group->pending() == 0
                &&
            group->is_cancelled()
                &&
            group->timed_out(),
            "v5.1 deadline: group cancel_after drains pending"
        );


        int cancelled_count = 0;

        std::future<int>* futures[] = {&a, &b};

        for(auto* f : futures)
        {
            try
            {
                (void)f->get();
            }
            catch(const TaskCancelled&)
            {
                ++cancelled_count;
            }
            catch(...)
            {
            }
        }

        expect(
            cancelled_count == 2
                &&
            group->errors().size() == 2,
            "v5.1 deadline: group futures carry TaskCancelled"
        );


        /* Never let a regression hang the whole test process. */
        if(!drained)
            group.release();

        pool.wait();
    }


    /* A future deadline leaves a group's work untouched. */
    {
        ThreadPool pool;

        TaskGroup group(pool);

        group.cancel_at(
            clock::now() + std::chrono::seconds(10)
        );

        auto f = group.submit(
            []()
            {
                return 5;
            }
        );

        expect(
            f.get() == 5
                &&
            !group.is_cancelled()
                &&
            !group.timed_out(),
            "v5.1 deadline: future group deadline is inert"
        );

        group.wait();
    }
}


int main()
{
    test_basics();

    test_priority();

    test_cancel();

    test_pause_keeps_workers();

    test_workers_are_reclaimed();

    test_wait_from_worker();

    test_default_handle();

    test_shutdown();

    test_clear_and_caller_runs();

    test_resize();

    test_runtime_config();

    test_scheduled_and_parallel();

    test_runtime_batch_and_accessors();

    test_wait_stress();

    test_scaling_churn();

    test_work_stealing_and_composition();

    test_v5_context_and_cancellation();

    test_v5_continuations_batches_and_groups();

    test_v501_discard_and_context();

    test_v510_deadlines();


    std::printf(
        "\n%s (%d failure(s))\n",
        failures == 0 ? "ALL PASS" : "FAILED",
        failures
    );


    return failures == 0 ? 0 : 1;
}
