::Test

::Framework.exe -c "Cube" -t 10 -mc 10 -f "Simulation_StarlinkFull.json"
::Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -s 5 -f "Simulation_StarlinkFull.json"


Framework.exe -c "Cube" -t 10 -mc 4 -f "Simulation_StarlinkFull_Collision.json"
Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -s 5 -f "Simulation_StarlinkFull_Collision.json"

Framework.exe -c "Cube" -t 10 -mc 10 -f "Simulation_StarlinkFull_Explosion.json"
Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -s 5 -f "Simulation_StarlinkFull_Explosion.json"