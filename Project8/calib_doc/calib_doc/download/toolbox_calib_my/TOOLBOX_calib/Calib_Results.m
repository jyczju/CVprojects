% Intrinsic and Extrinsic Camera Parameters
%
% This script file can be directly executed under Matlab to recover the camera intrinsic and extrinsic parameters.
% IMPORTANT: This file contains neither the structure of the calibration objects nor the image coordinates of the calibration points.
%            All those complementary variables are saved in the complete matlab data file Calib_Results.mat.
% For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/


%-- Focal length:
fc = [ 690.674615398043557 ; 691.219635807443638 ];

%-- Principal point:
cc = [ 337.332280627914770 ; 274.541670482865243 ];

%-- Skew coefficient:
alpha_c = 0.000000000000000;

%-- Distortion coefficients:
kc = [ 0.332535085643772 ; -0.863243246522823 ; 0.000722461914586 ; 0.001249460448016 ; 0.000000000000000 ];

%-- Focal length uncertainty:
fc_error = [ 3.889143736651819 ; 3.786285297745812 ];

%-- Principal point uncertainty:
cc_error = [ 2.428799157708886 ; 2.959049977604083 ];

%-- Skew coefficient uncertainty:
alpha_c_error = 0.000000000000000;

%-- Distortion coefficients uncertainty:
kc_error = [ 0.016033883931391 ; 0.063057889403018 ; 0.002207338223295 ; 0.001755893618106 ; 0.000000000000000 ];

%-- Image size:
nx = 640;
ny = 480;


%-- Various other variables (may be ignored if you do not use the Matlab Calibration Toolbox):
%-- Those variables are used to control which intrinsic parameters should be optimized

n_ima = 27;						% Number of calibration images
est_fc = [ 1 ; 1 ];					% Estimation indicator of the two focal variables
est_aspect_ratio = 1;				% Estimation indicator of the aspect ratio fc(2)/fc(1)
center_optim = 1;					% Estimation indicator of the principal point
est_alpha = 0;						% Estimation indicator of the skew coefficient
est_dist = [ 1 ; 1 ; 1 ; 1 ; 0 ];	% Estimation indicator of the distortion coefficients


%-- Extrinsic parameters:
%-- The rotation (omc_kk) and the translation (Tc_kk) vectors for every calibration image and their uncertainties

%-- Image #1:
omc_1 = [ 1.938296e+00 ; 2.022116e+00 ; 4.898874e-02 ];
Tc_1  = [ -9.289114e+01 ; -2.192292e+02 ; 1.558868e+03 ];
omc_error_1 = [ 4.053750e-03 ; 3.487103e-03 ; 7.472844e-03 ];
Tc_error_1  = [ 5.466598e+00 ; 6.613088e+00 ; 9.003115e+00 ];

%-- Image #2:
omc_2 = [ 2.182200e+00 ; 1.492847e+00 ; 1.433824e-01 ];
Tc_2  = [ -3.242724e+02 ; -1.544174e+02 ; 1.465086e+03 ];
omc_error_2 = [ 4.054081e-03 ; 2.748350e-03 ; 6.136381e-03 ];
Tc_error_2  = [ 5.189140e+00 ; 6.252094e+00 ; 8.849114e+00 ];

%-- Image #3:
omc_3 = [ 1.016530e+00 ; 2.856690e+00 ; -3.503305e-01 ];
Tc_3  = [ 7.225563e+01 ; -5.051620e+02 ; 1.931214e+03 ];
omc_error_3 = [ 2.562045e-03 ; 5.961347e-03 ; 9.532374e-03 ];
Tc_error_3  = [ 6.793322e+00 ; 8.171338e+00 ; 1.091943e+01 ];

%-- Image #4:
omc_4 = [ -2.018211e+00 ; -1.989788e+00 ; -5.166890e-01 ];
Tc_4  = [ 5.186627e+01 ; -2.435785e+02 ; 1.574420e+03 ];
omc_error_4 = [ 3.236062e-03 ; 4.648152e-03 ; 7.464567e-03 ];
Tc_error_4  = [ 5.582384e+00 ; 6.709785e+00 ; 9.842072e+00 ];

%-- Image #5:
omc_5 = [ -1.925553e+00 ; -1.945695e+00 ; 8.332917e-01 ];
Tc_5  = [ 1.682433e+02 ; -3.125162e+02 ; 2.035146e+03 ];
omc_error_5 = [ 4.258831e-03 ; 3.801117e-03 ; 7.588112e-03 ];
Tc_error_5  = [ 7.197817e+00 ; 8.644890e+00 ; 1.041254e+01 ];

%-- Image #6:
omc_6 = [ 2.053703e+00 ; 1.815594e+00 ; 8.879343e-02 ];
Tc_6  = [ -5.261244e+02 ; -3.392324e+02 ; 1.716143e+03 ];
omc_error_6 = [ 4.461470e-03 ; 4.048427e-03 ; 8.384628e-03 ];
Tc_error_6  = [ 6.104204e+00 ; 7.429113e+00 ; 1.063187e+01 ];

%-- Image #7:
omc_7 = [ -2.081628e+00 ; -2.145366e+00 ; 6.615466e-02 ];
Tc_7  = [ -7.576844e+01 ; -5.945344e+02 ; 1.900981e+03 ];
omc_error_7 = [ 8.244145e-03 ; 8.337063e-03 ; 1.426693e-02 ];
Tc_error_7  = [ 6.813244e+00 ; 8.135384e+00 ; 1.183772e+01 ];

%-- Image #8:
omc_8 = [ 2.189294e+00 ; 2.015667e+00 ; -4.883328e-01 ];
Tc_8  = [ -1.954892e+02 ; -1.239571e+02 ; 1.804382e+03 ];
omc_error_8 = [ 4.753741e-03 ; 4.287375e-03 ; 9.142895e-03 ];
Tc_error_8  = [ 6.289037e+00 ; 7.654742e+00 ; 9.809484e+00 ];

%-- Image #9:
omc_9 = [ 2.081133e+00 ; 2.002292e+00 ; -4.758984e-02 ];
Tc_9  = [ 1.778395e+02 ; -2.184647e+02 ; 2.789121e+03 ];
omc_error_9 = [ 6.173214e-03 ; 5.376632e-03 ; 1.171176e-02 ];
Tc_error_9  = [ 9.897273e+00 ; 1.189336e+01 ; 1.712975e+01 ];

%-- Image #10:
omc_10 = [ 1.767159e+00 ; 1.993393e+00 ; 2.357204e-01 ];
Tc_10  = [ -2.357615e+01 ; -2.963022e+02 ; 2.624405e+03 ];
omc_error_10 = [ 5.074127e-03 ; 4.215275e-03 ; 8.332296e-03 ];
Tc_error_10  = [ 9.269045e+00 ; 1.120252e+01 ; 1.549315e+01 ];

%-- Image #11:
omc_11 = [ 1.663520e+00 ; 1.851037e+00 ; 5.117168e-01 ];
Tc_11  = [ -3.224921e+01 ; -3.441474e+02 ; 2.567089e+03 ];
omc_error_11 = [ 4.644132e-03 ; 3.659506e-03 ; 6.772080e-03 ];
Tc_error_11  = [ 9.079420e+00 ; 1.096304e+01 ; 1.547957e+01 ];

%-- Image #12:
omc_12 = [ -1.953135e+00 ; -2.173541e+00 ; -8.436038e-01 ];
Tc_12  = [ 1.043455e+02 ; -1.364876e+02 ; 1.919461e+03 ];
omc_error_12 = [ 2.805257e-03 ; 5.154874e-03 ; 7.303588e-03 ];
Tc_error_12  = [ 6.805288e+00 ; 8.246210e+00 ; 1.248707e+01 ];

%-- Image #13:
omc_13 = [ -1.816021e+00 ; -1.960937e+00 ; 3.282999e-01 ];
Tc_13  = [ 2.250202e+02 ; -3.408230e+02 ; 2.339937e+03 ];
omc_error_13 = [ 5.230777e-03 ; 5.600668e-03 ; 1.063476e-02 ];
Tc_error_13  = [ 8.334988e+00 ; 9.976588e+00 ; 1.357360e+01 ];

%-- Image #14:
omc_14 = [ 1.936889e+00 ; 2.216437e+00 ; 1.661583e-02 ];
Tc_14  = [ 5.626498e+02 ; -2.962471e+02 ; 2.691425e+03 ];
omc_error_14 = [ 6.531911e-03 ; 4.970977e-03 ; 1.202801e-02 ];
Tc_error_14  = [ 9.608898e+00 ; 1.160109e+01 ; 1.612931e+01 ];

%-- Image #15:
omc_15 = [ -2.315523e+00 ; -2.119164e+00 ; 2.517779e-01 ];
Tc_15  = [ -1.028999e+03 ; -3.068716e+02 ; 2.567587e+03 ];
omc_error_15 = [ 7.203012e-03 ; 4.902544e-03 ; 1.275719e-02 ];
Tc_error_15  = [ 9.329800e+00 ; 1.151807e+01 ; 1.638667e+01 ];

%-- Image #16:
omc_16 = [ 2.089524e+00 ; 1.982707e+00 ; 5.696025e-01 ];
Tc_16  = [ -9.651974e+02 ; -8.531148e+02 ; 2.543156e+03 ];
omc_error_16 = [ 7.639157e-03 ; 1.188600e-02 ; 1.780042e-02 ];
Tc_error_16  = [ 9.469248e+00 ; 1.091306e+01 ; 2.051932e+01 ];

%-- Image #17:
omc_17 = [ 1.958840e+00 ; 2.008732e+00 ; 2.000854e-01 ];
Tc_17  = [ -1.050223e+03 ; 1.268355e+02 ; 2.654766e+03 ];
omc_error_17 = [ 1.275539e-02 ; 1.500393e-02 ; 2.638428e-02 ];
Tc_error_17  = [ 9.789852e+00 ; 1.163373e+01 ; 2.014705e+01 ];

%-- Image #18:
omc_18 = [ 2.232217e+00 ; 2.119446e+00 ; 1.838201e-01 ];
Tc_18  = [ 2.692337e+02 ; 1.552085e+02 ; 2.599826e+03 ];
omc_error_18 = [ 7.020261e-03 ; 4.358344e-03 ; 1.414009e-02 ];
Tc_error_18  = [ 9.212543e+00 ; 1.118919e+01 ; 1.648502e+01 ];

%-- Image #19:
omc_19 = [ 2.241309e+00 ; 2.144141e+00 ; 1.067090e-01 ];
Tc_19  = [ 3.177707e+02 ; -8.740739e+02 ; 2.722763e+03 ];
omc_error_19 = [ 8.906624e-03 ; 8.614037e-03 ; 1.398002e-02 ];
Tc_error_19  = [ 1.003490e+01 ; 1.172590e+01 ; 1.818768e+01 ];

%-- Image #20:
omc_20 = [ -1.797083e+00 ; -1.663258e+00 ; 8.631054e-01 ];
Tc_20  = [ -6.023291e+01 ; -3.201400e+01 ; 2.534785e+03 ];
omc_error_20 = [ 4.112272e-03 ; 3.528155e-03 ; 6.637248e-03 ];
Tc_error_20  = [ 8.953806e+00 ; 1.083576e+01 ; 1.266697e+01 ];

%-- Image #21:
omc_21 = [ 2.062289e+00 ; 1.909968e+00 ; 8.891583e-01 ];
Tc_21  = [ 9.766739e+01 ; -6.324671e+01 ; 2.194181e+03 ];
omc_error_21 = [ 4.897052e-03 ; 2.995085e-03 ; 7.154373e-03 ];
Tc_error_21  = [ 7.776710e+00 ; 9.408545e+00 ; 1.426237e+01 ];

%-- Image #22:
omc_22 = [ -2.121705e+00 ; -2.229364e+00 ; 3.394026e-01 ];
Tc_22  = [ -1.805563e+02 ; -2.842770e+02 ; 1.690661e+03 ];
omc_error_22 = [ 4.563634e-03 ; 4.844526e-03 ; 1.027639e-02 ];
Tc_error_22  = [ 5.907819e+00 ; 7.150321e+00 ; 9.547742e+00 ];

%-- Image #23:
omc_23 = [ 2.053946e+00 ; 2.098719e+00 ; -1.516716e-01 ];
Tc_23  = [ -2.423070e+02 ; -2.592136e+02 ; 2.207733e+03 ];
omc_error_23 = [ 6.865181e-03 ; 6.331364e-03 ; 1.384212e-02 ];
Tc_error_23  = [ 7.745471e+00 ; 9.394944e+00 ; 1.275688e+01 ];

%-- Image #24:
omc_24 = [ -2.443638e+00 ; -1.678136e+00 ; -7.638743e-01 ];
Tc_24  = [ -3.049358e+02 ; -3.033339e+02 ; 1.509991e+03 ];
omc_error_24 = [ 4.239698e-03 ; 4.128176e-03 ; 7.512433e-03 ];
Tc_error_24  = [ 5.394420e+00 ; 6.537193e+00 ; 9.844092e+00 ];

%-- Image #25:
omc_25 = [ 1.576120e+00 ; 1.963651e+00 ; 2.367739e-01 ];
Tc_25  = [ -3.715005e+02 ; -2.248605e+02 ; 1.737789e+03 ];
omc_error_25 = [ 3.633598e-03 ; 3.944512e-03 ; 6.423300e-03 ];
Tc_error_25  = [ 6.148058e+00 ; 7.490297e+00 ; 1.063710e+01 ];

%-- Image #26:
omc_26 = [ -2.581230e+00 ; -1.377341e+00 ; 6.573870e-01 ];
Tc_26  = [ -2.951759e+02 ; -4.313726e+01 ; 1.824543e+03 ];
omc_error_26 = [ 4.849741e-03 ; 2.271219e-03 ; 8.065051e-03 ];
Tc_error_26  = [ 6.385578e+00 ; 7.733918e+00 ; 9.205331e+00 ];

%-- Image #27:
omc_27 = [ 2.068497e+00 ; 2.158229e+00 ; -2.251310e-01 ];
Tc_27  = [ 4.269492e+01 ; -2.669245e+02 ; 2.292497e+03 ];
omc_error_27 = [ 6.123371e-03 ; 5.814072e-03 ; 1.255303e-02 ];
Tc_error_27  = [ 8.107699e+00 ; 9.735010e+00 ; 1.333374e+01 ];
