#include "threadpool.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>


using namespace std::chrono_literals;


int main()
{
    ThreadPool::Config cfg;

    cfg.min_threads = 2;
    cfg.max_threads = 6;

    cfg.max_queue = 20;

    cfg.idle_timeout = 2s;

    cfg.reject = RejectPolicy::BLOCK;


    /*
     * Optional hook: invoked once per worker thread.
     */
    std::atomic<int> workers_started{0};


    cfg.thread_init =
        [&workers_started]()
        {
            int index =
                workers_started.fetch_add(
                    1
                ) + 1;


            std::cout
                << "[init] worker #"
                << index
                << " ready\n";
        };


    ThreadPool pool(cfg);


    pool.set_reject_callback(
        []()
        {
            std::cerr
                << "[reject] task rejected\n";
        }
    );


    // ========================================================
    // Normal task
    // ========================================================

    auto f1 =
        pool.submit(
            []()
            {
                std::this_thread::sleep_for(
                    300ms
                );

                return 42;
            }
        );


    std::cout
        << "f1 = "
        << f1.get()
        << '\n';


    // ========================================================
    // Task with arguments
    // ========================================================

    auto f2 =
        pool.submit(
            [](int a, int b)
            {
                return a + b;
            },
            10,
            20
        );


    std::cout
        << "10 + 20 = "
        << f2.get()
        << '\n';


    // ========================================================
    // Exception propagation through std::future
    // ========================================================

    auto error_future =
        pool.submit(
            []() -> int
            {
                throw std::runtime_error(
                    "task error"
                );
            }
        );


    try
    {
        error_future.get();
    }
    catch(const std::exception& e)
    {
        std::cout
            << "future exception: "
            << e.what()
            << '\n';
    }


    // ========================================================
    // Priority tasks
    // ========================================================

    std::vector<
        std::future<void>
    > futures;


    for(int i = 0; i < 10; ++i)
    {
        futures.emplace_back(
            pool.submit_priority(
                i,
                [i]()
                {
                    std::cout
                        << "priority task: "
                        << i
                        << '\n';


                    std::this_thread::sleep_for(
                        100ms
                    );
                }
            )
        );
    }


    for(auto& f : futures)
    {
        f.get();
    }


    // ========================================================
    // Cancellable task
    // ========================================================

    auto cancellable =
        pool.submit_cancellable(
            5,
            []()
            {
                std::this_thread::sleep_for(
                    1s
                );

                return std::string(
                    "finished"
                );
            }
        );


    /*
     * cancel() returns true:
     * the WAITING -> CANCELLED transition succeeded.
     *
     * cancel() returns false:
     * the task has probably already started or completed.
     */
    bool cancelled =
        cancellable.handle.cancel();


    std::cout
        << "cancel result = "
        << std::boolalpha
        << cancelled
        << '\n';


    /*
     * If the packaged_task is never executed, the associated future may
     * eventually report std::future_error with broken_promise.
     *
     * Therefore, cancelled futures should generally be handled with
     * exception handling.
     */
    try
    {
        auto value =
            cancellable.future.get();


        std::cout
            << value
            << '\n';
    }
    catch(const std::exception& e)
    {
        std::cout
            << "cancelled future: "
            << e.what()
            << '\n';
    }


    // ========================================================
    // Move-only argument
    //
    // The callable and its arguments are forwarded, so std::unique_ptr
    // works as an argument.
    // ========================================================

    auto move_only =
        pool.submit(
            [](std::unique_ptr<int> value)
            {
                return *value;
            },
            std::make_unique<int>(99)
        );


    std::cout
        << "move-only arg = "
        << move_only.get()
        << '\n';


    // ========================================================
    // Pause / resume with non-throwing submission
    //
    // try_submit() never waits for a free slot and returns
    // std::nullopt instead of throwing.
    // ========================================================

    pool.pause();


    std::vector<
        std::future<int>
    > paused_batch;


    std::size_t queued = 0;


    for(int i = 0;
        i < 40;
        ++i)
    {
        auto maybe =
            pool.try_submit(
                [i]()
                {
                    return i;
                }
            );


        if(!maybe)
        {
            std::cout
                << "try_submit rejected at #"
                << i
                << '\n';

            break;
        }


        paused_batch.emplace_back(
            std::move(*maybe)
        );


        ++queued;
    }


    std::cout
        << "queued while paused = "
        << queued
        << '\n';


    pool.resume();


    for(auto& f : paused_batch)
    {
        f.get();
    }


    /*
     * wait() returns once the queue is empty and every task finished.
     */
    pool.wait();


    {
        auto m =
            pool.get_metrics();


        std::cout
            << "\n=== after wait ===\n";

        std::cout
            << "active    : "
            << m.active
            << '\n';

        std::cout
            << "completed : "
            << m.completed
            << '\n';
    }


    // ========================================================
    // Submit many tasks to test dynamic expansion
    // ========================================================

    std::vector<
        std::future<int>
    > batch;


    for(int i = 0; i < 30; ++i)
    {
        batch.emplace_back(
            pool.submit(
                [i]()
                {
                    std::this_thread::sleep_for(
                        200ms
                    );

                    return i * i;
                }
            )
        );
    }


    /*
     * Inspect the current thread-pool metrics.
     */
    std::this_thread::sleep_for(
        100ms
    );


    {
        auto m =
            pool.get_metrics();


        std::cout
            << "\n=== metrics ===\n";

        std::cout
            << "workers   : "
            << m.worker_count
            << '\n';

        std::cout
            << "idle      : "
            << m.idle_count
            << '\n';

        std::cout
            << "queue     : "
            << m.queue_size
            << '\n';

        std::cout
            << "running   : "
            << m.running
            << '\n';

        std::cout
            << "submitted : "
            << m.submitted
            << '\n';

        std::cout
            << "completed : "
            << m.completed
            << '\n';

        std::cout
            << "cancelled : "
            << m.cancelled
            << '\n';

        std::cout
            << "rejected  : "
            << m.rejected
            << '\n';
    }


    for(auto& f : batch)
    {
        try
        {
            f.get();
        }
        catch(...)
        {
        }
    }


    /*
     * Wait long enough to observe dynamic shrinking.
     */
    std::this_thread::sleep_for(
        3s
    );


    {
        auto m =
            pool.get_metrics();


        std::cout
            << "\n=== after idle ===\n";

        std::cout
            << "workers : "
            << m.worker_count
            << '\n';

        std::cout
            << "idle    : "
            << m.idle_count
            << '\n';
    }


    // ========================================================
    // RejectPolicy::CALLER_RUNS
    //
    // A saturated pool executes the task in the submitting thread
    // instead of blocking or discarding it.
    // ========================================================

    ThreadPool::Config caller_cfg;

    caller_cfg.min_threads = 1;
    caller_cfg.max_threads = 1;

    caller_cfg.max_queue = 1;

    caller_cfg.idle_timeout = 500ms;

    caller_cfg.reject =
        RejectPolicy::CALLER_RUNS;


    ThreadPool caller_pool(caller_cfg);


    std::vector<
        std::future<std::thread::id>
    > ids;


    for(int i = 0; i < 6; ++i)
    {
        ids.emplace_back(
            caller_pool.submit(
                []()
                {
                    std::this_thread::sleep_for(
                        150ms
                    );

                    return std::this_thread::get_id();
                }
            )
        );
    }


    int caller_ran = 0;


    for(auto& f : ids)
    {
        if(f.get()
            == std::this_thread::get_id())
        {
            ++caller_ran;
        }
    }


    std::cout
        << "\ntasks executed in the caller thread = "
        << caller_ran
        << '\n';


    // ========================================================
    // Graceful shutdown
    // ========================================================

    pool.stop(
        PoolStopMode::GRACEFUL
    );


    std::cout
        << "\npool stopped\n";


    return 0;
}