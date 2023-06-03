#!/bin/bash

# Bash-Skript zum wiederholten Ausführen des Skripts add_to_data.py mit Schritten von 1000 für begin und end.

# Schleife über die Werte von 0 bis 27000 in Schritten von 1000
for ((i = 0; i <= 27000; i += 1000))
do
    begin=$i
    end=$((i + 1000))
    python add_to_data.py $begin $end  # Führe add_to_data.py mit den aktuellen Werten von begin und end aus
done