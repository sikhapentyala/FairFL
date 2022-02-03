#!/bin/bash


python server_ads.py &
sleep 2 # Sleep for 2s to give the server enough time to start
#users='91'
#$users=[6,92,110,8,84,82,3,48,79,61,88,87,86,50,78,32,30,80,67,11,103,27,71,56,95,116,44,111,93,100,109,41,99,16,22,26,117,15,49,9,112,21,51,54,35,45,118,72,85,5,115,113,10,114,60,70,66,89,63,81,38,83,7,46,29,55,13,68,31,77,101,17,62,36,28,12,53,40,37,47,20,105,75,97,2,59,64,23,90,42,98,24,4,108,19,58,33,119,39,120,94,96,76,43,102,73,74,18,1,34,14,65,107]
users="6 92 110 8 84 82 3 48 79 61 88 87 86 50 78 32 30 80 67 11 103 27 71 56 95 116 44 111 93 100 109 41 99 16 22 26 117 15 49 9 112 21 51 54 35 45 118 72 85 5 115 113 10 114 60 70 66 89 63 81 38 83 7 46 29 55 13 68 31 77 101 17 62 36 28 12 53 40 37 47 20 105 75 97 2 59 64 23 90 42 98 24 4 108 19 58 33 119 39 120 94 96 76 43 102 73 74 18 1 34 14 65 107"
#for i in `seq 1 120`; do
for i in $users
do
    # shellcheck disable=SC1072
    echo "Starting client $i"
    python client_ads.py --partition=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait


