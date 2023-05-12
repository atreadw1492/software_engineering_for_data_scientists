

import argparse
 
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
#parser.add_argument("-c", "--config")
 
# Read arguments from command line
args = parser.parse_args()

print(args.config) 
#print(args)
#if args.config:
#    print("Displaying Output as: % s" % args.config)