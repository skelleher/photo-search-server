import os
import argparse
from collections import namedtuple

Result = namedtuple("Result", "classname, total, top_1, top_5")
results = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="score.log to summarize")
    
    args = parser.parse_args()
    
    # Check if input path exists.
    if not os.path.exists(args.input):
        print("Error: %s not found" % args.input)
        return -1
    
    print("Scanning %s..." % args.input)
    
    with open(args.input, "rt") as logfile:
        while (True):
            folder = logfile.readline()
            prev   = logfile.tell()
            total  = logfile.readline()

            if not total:
                break
                
            if "Total" not in total:
                logfile.seek(prev)
                continue
   
            folder = folder.rstrip()
            total = total.rstrip()
    
            elements  = folder.split(os.sep)
            classname = elements[-1]
            #print("class = %s: %s" % (classname, total))

            totals = total.split(" ")
            count = totals[2]
            top_1 = totals[5]
            top_5 = totals[8]

            top_1 = (100.0 * (float(top_1) / float(count)))
            top_5 = (100.0 * (float(top_5) / float(count)))

            #print("top-1: %f" % top_1)
            #print("top-5: %f" % top_5)

            result = Result(classname, count, top_1, top_5)

            results.append(result)   


    num_classes = len(results)
    total_top_1_accuracy = 0.0
    total_top_5_accuracy = 0.0

    for result in results:
        total_top_1_accuracy += result.top_1
        total_top_5_accuracy += result.top_5

    print("top-1 mAP: %f top-5 mAP: %f" % (total_top_1_accuracy/num_classes, total_top_5_accuracy/num_classes))

    def sortKey(result):
        return result.top_1

    top = sorted(results, key = sortKey, reverse = True)
    print("Best 5 classes:")
    for i in range(5):
        print(top[i])

    print("Worst 5 classes:")
    for i in xrange(1,6):
        print(top[ len(top) - i ])


if __name__ == "__main__":
    main()
