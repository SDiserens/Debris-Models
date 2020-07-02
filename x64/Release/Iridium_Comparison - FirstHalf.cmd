::Test
Framework.exe -c "Cube" -t 1 -f "Collisions_Iridium_first.json"
Framework.exe -c "Cube" -t 2 -f "Collisions_Iridium_first.json"
Framework.exe -c "Cube" -t 5 -f "Collisions_Iridium_first.json"
Framework.exe -c "Cube" -t 10 -f "Collisions_Iridium_first.json"
Framework.exe -c "Cube" -t 20 -f "Collisions_Iridium_first.json"

::Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_Iridium_first.json"
