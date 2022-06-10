% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 685.686491944376144 ; 685.679325544727476 ];

%-- Principal point:
cc = [ 342.199678804392363 ; 278.134653797208614 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.309453372407608 ; -0.755748739831903 ; 0.002717630730168 ; 0.003646681528820 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 5.830385871753291 ; 5.764095867012058 ];

%-- Principal point uncertainty:
cc_error = [ 4.236906898835468 ; 4.258982721463696 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.023904761520276 ; 0.092309691292127 ; 0.002854529414585 ; 0.002895101591707 ; 0.000000000000000 ];

%-- Image size:
nx = 640;
ny = 480;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 10;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ 1.939516e+00 ; 2.020085e+00 ; 5.634778e-02 ];
Tc_1  = [ -1.990387e+02 ; -3.317635e+02 ; 1.539380e+03 ];
omc_error_1 = [ 5.558425e-03 ; 4.957600e-03 ; 9.795804e-03 ];
Tc_error_1  = [ 9.505257e+00 ; 9.418950e+00 ; 1.339958e+01 ];

%-- Image #2:
omc_2 = [ 2.180179e+00 ; 1.488618e+00 ; 1.496133e-01 ];
Tc_2  = [ -4.578907e+02 ; -2.240562e+02 ; 1.423161e+03 ];
omc_error_2 = [ 5.617193e-03 ; 4.106950e-03 ; 8.370084e-03 ];
Tc_error_2  = [ 8.925599e+00 ; 8.805633e+00 ; 1.295969e+01 ];

%-- Image #3:
omc_3 = [ 1.016325e+00 ; 2.852418e+00 ; -3.408716e-01 ];
Tc_3  = [ 7.263751e+01 ; -6.517841e+02 ; 1.946288e+03 ];
omc_error_3 = [ 2.866996e-03 ; 7.630921e-03 ; 1.123962e-02 ];
Tc_error_3  = [ 1.202132e+01 ; 1.181816e+01 ; 1.671802e+01 ];

%-- Image #4:
omc_4 = [ -2.017575e+00 ; -1.991920e+00 ; -5.220314e-01 ];
Tc_4  = [ -5.860243e+01 ; -3.398080e+02 ; 1.512054e+03 ];
omc_error_4 = [ 4.283575e-03 ; 6.481965e-03 ; 1.061614e-02 ];
Tc_error_4  = [ 9.416268e+00 ; 9.334669e+00 ; 1.373116e+01 ];

%-- Image #5:
omc_5 = [ -1.924956e+00 ; -1.945271e+00 ; 8.215928e-01 ];
Tc_5  = [ 7.866624e+01 ; -4.162527e+02 ; 2.093671e+03 ];
omc_error_5 = [ 6.312765e-03 ; 5.250535e-03 ; 1.042374e-02 ];
Tc_error_5  = [ 1.300951e+01 ; 1.278221e+01 ; 1.548972e+01 ];

%-- Image #6:
omc_6 = [ 2.052754e+00 ; 1.811809e+00 ; 1.046457e-01 ];
Tc_6  = [ -6.466624e+02 ; -4.369722e+02 ; 1.680067e+03 ];
omc_error_6 = [ 5.776169e-03 ; 5.713993e-03 ; 1.048757e-02 ];
Tc_error_6  = [ 1.058346e+01 ; 1.060493e+01 ; 1.583458e+01 ];

%-- Image #7:
omc_7 = [ -2.083688e+00 ; -2.144650e+00 ; 5.268269e-02 ];
Tc_7  = [ -1.862408e+02 ; -7.067575e+02 ; 1.885492e+03 ];
omc_error_7 = [ 9.139518e-03 ; 9.151965e-03 ; 1.500572e-02 ];
Tc_error_7  = [ 1.190203e+01 ; 1.166728e+01 ; 1.724933e+01 ];

%-- Image #8:
omc_8 = [ 2.191746e+00 ; 2.016424e+00 ; -4.741471e-01 ];
Tc_8  = [ -3.129251e+02 ; -2.182002e+02 ; 1.830034e+03 ];
omc_error_8 = [ 5.480400e-03 ; 5.374363e-03 ; 1.142499e-02 ];
Tc_error_8  = [ 1.112944e+01 ; 1.119238e+01 ; 1.492050e+01 ];

%-- Image #9:
omc_9 = [ 2.081855e+00 ; 1.999554e+00 ; -4.093870e-02 ];
Tc_9  = [ 5.366631e+01 ; -3.283286e+02 ; 2.766498e+03 ];
omc_error_9 = [ 7.704569e-03 ; 6.823678e-03 ; 1.455948e-02 ];
Tc_error_9  = [ 1.714026e+01 ; 1.701126e+01 ; 2.494962e+01 ];

%-- Image #10:
omc_10 = [ 1.767161e+00 ; 1.990016e+00 ; 2.487575e-01 ];
Tc_10  = [ -1.239744e+02 ; -4.232879e+02 ; 2.575518e+03 ];
omc_error_10 = [ 6.544748e-03 ; 5.815136e-03 ; 1.050727e-02 ];
Tc_error_10  = [ 1.599221e+01 ; 1.589911e+01 ; 2.317434e+01 ];

