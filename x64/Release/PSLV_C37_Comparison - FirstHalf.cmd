::Test
Framework.exe -c "Cube" -t 1 -f "Collisions_PSLV_C37_first.json"
Framework.exe -c "Cube" -t 2 -f "Collisions_PSLV_C37_first.json"
Framework.exe -c "Cube" -t 5 -f "Collisions_PSLV_C37_first.json"
Framework.exe -c "Cube" -t 10 -f "Collisions_PSLV_C37_first.json"
Framework.exe -c "Cube" -t 20 -f "Collisions_PSLV_C37_first.json"
Framework.exe -c "Cube" -t 50 -f "Collisions_PSLV_C37_first.json"
Framework.exe -c "Cube" -t 100 -f "Collisions_PSLV_C37_first.json"

Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_PSLV_C37_first.json"
