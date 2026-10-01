"""
Zepto Data & AI Platform - Interactive Project Runner / Demo
Author: Manikanta Reddy Reddygari
"""

import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_module_1():
    print("\n" + "=" * 70)
    print(">>> RUNNING MODULE 1: DATA PIPELINE")
    print("=" * 70)
    script = os.path.join(PROJECT_ROOT, "data_pipeline", "run_pipeline.py")
    subprocess.run([sys.executable, script], check=True)


def run_module_2():
    print("\n" + "=" * 70)
    print(">>> RUNNING MODULE 2: ANALYTICS PIPELINE")
    print("=" * 70)
    print("\n--- Part A: Exploratory Data Analysis & Chart Generation ---")
    eda_script = os.path.join(PROJECT_ROOT, "analytics", "eda_pipeline.py")
    subprocess.run([sys.executable, eda_script], check=True)

    print("\n--- Part B: Predictive Modeling, Imbalance & Tuning ---")
    mod_script = os.path.join(PROJECT_ROOT, "analytics", "modeling_pipeline.py")
    subprocess.run([sys.executable, mod_script], check=True)

    print("\n--- Verification: Testing Inference on Raw Unprocessed Data ---")
    inf_script = os.path.join(PROJECT_ROOT, "analytics", "test_inference.py")
    subprocess.run([sys.executable, inf_script], check=True)


def run_module_3_tests():
    print("\n" + "=" * 70)
    print(">>> RUNNING MODULE 3: SUPPORT ASSISTANT TEST SUITE")
    print("=" * 70)
    test_script = os.path.join(PROJECT_ROOT, "support_assistant", "test_assistant.py")
    subprocess.run([sys.executable, test_script], check=True)


def run_module_3_chat():
    print("\n" + "=" * 70)
    print(">>> MODULE 3: INTERACTIVE SUPPORT ASSISTANT CHAT")
    print("=" * 70)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "support_assistant"))
    from graph import query_assistant

    print("Welcome to the Zepto Policy Support Assistant!")
    print("Try asking questions like:")
    print("  1. 'What is the delivery fee for orders below 149?'")
    print("  2. 'Can I return damaged grocery items?'")
    print("  3. 'How much does Zepto Pass cost?'")
    print("  4. 'What is the capital of France?' (Non-policy question)")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting chat demo. Goodbye!")
                break

            resp = query_assistant(user_input)
            print(f"\nZepto Assistant: {resp.answer}")
            print(f"Sources Cited: {resp.sources}")
            print(f"Confidence: {resp.confidence}\n" + "-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat demo.")
            break


def serve_api():
    print("\n" + "=" * 70)
    print(">>> STARTING FASTAPI SERVER (UVICORN)")
    print("=" * 70)
    print("Interactive Swagger UI: http://127.0.0.1:7860/docs")
    print("Health Check:           http://127.0.0.1:7860/health")
    print("Press Ctrl+C to stop the server.\n")
    import uvicorn
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "support_assistant"))
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=False)


def main():
    parser = argparse.ArgumentParser(description="Zepto Data & AI Platform Demo Runner")
    parser.add_argument("--module", choices=["1", "2", "3"], help="Run a specific module (1, 2, or 3)")
    parser.add_argument("--chat", action="store_true", help="Launch interactive terminal chat with Support Assistant")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI web server on port 7860")
    parser.add_argument("--all", action="store_true", help="Run the entire platform end-to-end")

    args = parser.parse_args()

    if args.module == "1":
        run_module_1()
    elif args.module == "2":
        run_module_2()
    elif args.module == "3":
        run_module_3_tests()
    elif args.chat:
        run_module_3_chat()
    elif args.serve:
        serve_api()
    elif args.all:
        run_module_1()
        run_module_2()
        run_module_3_tests()
    else:
        # Interactive menu
        print("=" * 70)
        print("  ZEPTO DATA & AI PLATFORM - DEMO RUNNER")
        print("  Author: Manikanta Reddy Reddygari")
        print("=" * 70)
        print("Select what you would like to run:")
        print("  1. Run Module 1: Data Pipeline (Scrape -> Clean -> SQLite -> SQL & Merge)")
        print("  2. Run Module 2: Analytics Pipeline (EDA -> Models -> SMOTE -> Inference)")
        print("  3. Run Module 3: Support Assistant (Automated Test Suite)")
        print("  4. Interactive Chat with Support Assistant in Terminal")
        print("  5. Start FastAPI Server & Swagger UI (http://127.0.0.1:7860/docs)")
        print("  6. Run Entire Project (Modules 1, 2, and 3 in sequence)")
        print("  0. Exit")
        print("=" * 70)

        choice = input("Enter choice (0-6): ").strip()
        if choice == "1":
            run_module_1()
        elif choice == "2":
            run_module_2()
        elif choice == "3":
            run_module_3_tests()
        elif choice == "4":
            run_module_3_chat()
        elif choice == "5":
            serve_api()
        elif choice == "6":
            run_module_1()
            run_module_2()
            run_module_3_tests()
        else:
            print("Exiting.")


if __name__ == "__main__":
    main()
