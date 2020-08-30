::Test

Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_Trad2006_1.json"

Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_Trad2006_2.json"
Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_Trad2006_R.json"
Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_Trad2015.json"

Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_Iridium2006.json"
::Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_Iridium_first.json"
Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_PSLV_C37_first.json"

Framework.exe -c "OrbitTrace" -t 0.0 -mc 1 -f "Collisions_IridiumNext.json"