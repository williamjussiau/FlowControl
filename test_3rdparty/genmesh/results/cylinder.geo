// Gmsh project created on Thu Apr 08 09:18:15 2021
SetFactory("OpenCASCADE");
//+
xinfa = DefineNumber[ -10, Name "Parameters/xinfa" ];
//+
xinf = DefineNumber[ 20, Name "Parameters/xinf" ];
//+
yinf = DefineNumber[ 8, Name "Parameters/yinf" ];
//+
xplus = DefineNumber[ 5, Name "Parameters/xplus" ];
//+
n1 = DefineNumber[ 24, Name "Parameters/n1" ];
//+
n2 = DefineNumber[ 11, Name "Parameters/n2" ];
//+
n3 = DefineNumber[ 1, Name "Parameters/n3" ];
//+
Point(1) = {xinfa, yinf, 0, 1.0};
//+
Point(2) = {xinfa, -yinf, 0, 1.0};
//+
Point(3) = {xinf, -yinf, 0, 1.0};
//+
Point(4) = {xinf, yinf, 0, 1.0};
//+
Point(5) = {xinf, 3, 0, 1.0};
//+
Point(6) = {xinf, -3, 0, 1.0};
//+
Point(7) = {xinfa, -3, 0, 1.0};
//+
Point(8) = {xinfa, 3, 0, 1.0};
//+
Point(9) = {-0.5, 0, 0, 1.0};
//+
Point(10) = {0.5, 0, 0, 1.0};
//+
Point(11) = {0, 0, 0, 1.0};
//+
Point(12) = {-1.5, 1.5, -0, 1.0};
//+
Point(13) = {-1.5, -1.5, -0, 1.0};
//+
Point(14) = {xplus, -1.5, -0, 1.0};
//+
Point(15) = {xplus, 1.5, -0, 1.0};
//+
Circle(1) = {-0, -0, 0, 0.5, 0, 2*Pi};
//+
Line(2) = {12, 13};
//+
Line(3) = {13, 14};
//+
Line(4) = {14, 15};
//+
Line(5) = {15, 12};
//+
Line(6) = {8, 7};
//+
Line(7) = {7, 6};
//+
Line(8) = {6, 5};
//+
Line(9) = {5, 8};
//+
Line(10) = {8, 1};
//+
Line(11) = {1, 4};
//+
Line(12) = {4, 5};
//+
Line(13) = {6, 3};
//+
Line(14) = {3, 2};
//+
Line(15) = {2, 7};
//+
Curve Loop(1) = {10, 11, 12, 9};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {15, 7, 13, 14};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {5, 2, 3, 4};
//+
Curve Loop(4) = {1};
//+
Plane Surface(3) = {3, 4};
//+
Curve Loop(5) = {9, 6, 7, 8};
//+
Curve Loop(6) = {5, 2, 3, 4};
//+
Plane Surface(4) = {5, 6};
//+
MeshSize {12, 15, 14, 13, 9, 10} = 1/n1;
//+
MeshSize {8, 7, 6, 5} = 1/n2;
//+
MeshSize {3, 4, 1, 2} = 1/n3;
