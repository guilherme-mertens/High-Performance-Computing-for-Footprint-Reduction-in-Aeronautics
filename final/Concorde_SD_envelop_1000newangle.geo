SetFactory("OpenCASCADE");
Merge "Concorde_skin_coarse1000.stp";
//+
Rotate {{1,0,0}, {0, 0, 0}, 0.5235987755982988} { Volume{1}; }
//+
Box(2) = {-150, -150, -150, 300, 300, 300};
//+
BooleanDifference(3) = { Volume{2};Delete;  }{ Volume{1}; Delete;};


//+
Physical Volume("fluid", 1900) = {3};
//+
StartSurface = 7;
EndSurface = 1156;

// Create an array to hold your surfaces
SurfaceArray[] = {};

// Use a For loop to populate the array
For i In {StartSurface:EndSurface}
    SurfaceArray[i-StartSurface] = i;
EndFor
// Create the physical surface with all the surfaces in the range
Physical Surface("plane", 1800) = SurfaceArray[];

//+
Physical Surface("inlet", 1500) = {3};
//+
Physical Surface("outlet", 1600) = {5};
//+
Physical Surface("wall", 1700) = {1,2,4,6};




//+
Mesh 3;


