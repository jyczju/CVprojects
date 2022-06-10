% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 682.352406775788268 ; 682.679680476819954 ];

%-- Principal point:
cc = [ 341.405572252971979 ; 276.989670077618314 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.328544765542388 ; -0.811755496715050 ; 0.001848322518921 ; 0.003623129009389 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 4.896908089223792 ; 4.843017821739185 ];

%-- Principal point uncertainty:
cc_error = [ 3.480926985970743 ; 3.425775266184014 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.020328850227824 ; 0.078477013554070 ; 0.002335671563587 ; 0.002416674144568 ; 0.000000000000000 ];

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
omc_1 = [ 1.938899e+00 ; 2.020512e+00 ; 5.541803e-02 ];
Tc_1  = [ -1.971018e+02 ; -3.293075e+02 ; 1.534462e+03 ];
omc_error_1 = [ 4.566755e-03 ; 4.099755e-03 ; 8.117386e-03 ];
Tc_error_1  = [ 7.815171e+00 ; 7.579010e+00 ; 1.125043e+01 ];

%-- Image #2:
omc_2 = [ 2.180730e+00 ; 1.490761e+00 ; 1.478599e-01 ];
Tc_2  = [ -4.568800e+02 ; -2.216374e+02 ; 1.418975e+03 ];
omc_error_2 = [ 4.587043e-03 ; 3.393170e-03 ; 6.911969e-03 ];
Tc_error_2  = [ 7.345521e+00 ; 7.088443e+00 ; 1.088469e+01 ];

%-- Image #3:
omc_3 = [ 1.015701e+00 ; 2.851540e+00 ; -3.434672e-01 ];
Tc_3  = [ 7.478546e+01 ; -6.485196e+02 ; 1.941066e+03 ];
omc_error_3 = [ 2.409078e-03 ; 6.321769e-03 ; 9.295785e-03 ];
Tc_error_3  = [ 9.883873e+00 ; 9.509495e+00 ; 1.400756e+01 ];

%-- Image #4:
omc_4 = [ -2.018692e+00 ; -1.992194e+00 ; -5.204558e-01 ];
Tc_4  = [ -5.690438e+01 ; -3.370707e+02 ; 1.506340e+03 ];
omc_error_4 = [ 3.528419e-03 ; 5.337918e-03 ; 8.795557e-03 ];
Tc_error_4  = [ 7.739957e+00 ; 7.512385e+00 ; 1.151085e+01 ];

%-- Image #5:
omc_5 = [ -1.924401e+00 ; -1.944368e+00 ; 8.240525e-01 ];
Tc_5  = [ 8.137859e+01 ; -4.128731e+02 ; 2.089773e+03 ];
omc_error_5 = [ 5.243156e-03 ; 4.405729e-03 ; 8.569717e-03 ];
Tc_error_5  = [ 1.070948e+01 ; 1.030099e+01 ; 1.300325e+01 ];

%-- Image #6:
omc_6 = [ 2.053083e+00 ; 1.812310e+00 ; 1.010799e-01 ];
Tc_6  = [ -6.450657e+02 ; -4.340129e+02 ; 1.676128e+03 ];
omc_error_6 = [ 4.807331e-03 ; 4.763321e-03 ; 8.729871e-03 ];
Tc_error_6  = [ 8.707876e+00 ; 8.539839e+00 ; 1.328711e+01 ];

%-- Image #7:
omc_7 = [ -2.079941e+00 ; -2.144734e+00 ; 6.287771e-02 ];
Tc_7  = [ -1.836391e+02 ; -7.046002e+02 ; 1.884376e+03 ];
omc_error_7 = [ 7.662770e-03 ; 7.681841e-03 ; 1.248331e-02 ];
Tc_error_7  = [ 9.807029e+00 ; 9.415754e+00 ; 1.447706e+01 ];

%-- Image #8:
omc_8 = [ 2.191270e+00 ; 2.015732e+00 ; -4.744117e-01 ];
Tc_8  = [ -3.109952e+02 ; -2.152761e+02 ; 1.824717e+03 ];
omc_error_8 = [ 4.592591e-03 ; 4.457529e-03 ; 9.496144e-03 ];
Tc_error_8  = [ 9.152136e+00 ; 9.013273e+00 ; 1.254495e+01 ];

%-- Image #9:
omc_9 = [ 2.080886e+00 ; 1.999360e+00 ; -4.069405e-02 ];
Tc_9  = [ 5.717895e+01 ; -3.238438e+02 ; 2.757099e+03 ];
omc_error_9 = [ 6.395981e-03 ; 5.674889e-03 ; 1.216876e-02 ];
Tc_error_9  = [ 1.409629e+01 ; 1.369293e+01 ; 2.097909e+01 ];

%-- Image #10:
omc_10 = [ 1.767200e+00 ; 1.991146e+00 ; 2.459233e-01 ];
Tc_10  = [ -1.209347e+02 ; -4.190219e+02 ; 2.566959e+03 ];
omc_error_10 = [ 5.418300e-03 ; 4.837202e-03 ; 8.734267e-03 ];
Tc_error_10  = [ 1.315413e+01 ; 1.279764e+01 ; 1.946869e+01 ];

