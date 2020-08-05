::Test

Framework.exe -c "Cube" -t 10 -f "Simulation_StarlinkFull.json"
::Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Simulation_StarlinkFull.json"