::Test

Framework.exe -c "Cube" -t 10 -mc 5 -f "Simulation_StarlinkFull.json"
Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -s 5 -f "Simulation_StarlinkFull.json"