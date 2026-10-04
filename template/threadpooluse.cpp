#include "threadpool.hpp"

#include <chrono>
#include <iostream>
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
    // Graceful shutdown
    // ========================================================

    pool.stop(
        PoolStopMode::GRACEFUL
    );


    std::cout
        << "\npool stopped\n";


    return 0;
}