
import stopit

with stopit.ThreadingTimeout(5) as context_manager:

    for i in range(10**8):
        i = i * 2

if context_manager.state == context_manager.EXECUTED:
    print("COMPLETE...")


elif context_manager.state == context_manager.TIMED_OUT:
    print("DID NOT FINISH...")

    raise AssertionError("DID NOT FINISH")