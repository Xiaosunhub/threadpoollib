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


    std::printf(
        "\n%s (%d failure(s))\n",
        failures == 0 ? "ALL PASS" : "FAILED",
        failures
    );


    return failures == 0 ? 0 : 1;
}
