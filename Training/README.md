The final models A & B were trained using complete pre processed Geuvadis Dataset. We used Huber loss for comparison among these 2 models and as expected model B performed better when trained on the complete dataset.

The final output of the script provided is as follows:
>>> OFFICIAL ARCHITECTURE REPLICATION | Device: cuda
>>> Loading Data Manager...
   Indexed 455 valid samples.
   Caching Expression Data...
   Parsing GTF...
   Calculating Whole-Population Z-Scores...
   Final Valid Genes: 200116

==================================================
[MODEL A] Paper Copy: 49152bp | MSE | Unfrozen
==================================================
   Ep 1: Train=2.0784 | Val=1.9613
   Ep 2: Train=1.9569 | Val=1.8410
   Ep 3: Train=1.9322 | Val=2.0030
   Ep 4: Train=1.8443 | Val=1.9818
   Ep 5: Train=1.8711 | Val=2.0158
   Ep 6: Train=1.9904 | Val=1.7292
   Ep 7: Train=1.9733 | Val=1.8226
   Ep 8: Train=1.9135 | Val=1.9579
   Ep 9: Train=1.9187 | Val=2.0300
   Ep 10: Train=1.9039 | Val=1.8305
Model A Final Score (MSE): 1.72921

==================================================
[MODEL B] Enhanced: 196608bp | SwiGLU | Huber
==================================================
   >>> Fold 1/5...
   Ep 1: Train=0.5263 | Val=0.4729
   Ep 2: Train=0.4955 | Val=0.4958
   Ep 3: Train=0.5099 | Val=0.4921
   Ep 4: Train=0.5053 | Val=0.4781
   Ep 5: Train=0.5145 | Val=0.5084
   Ep 6: Train=0.5059 | Val=0.5089
   Ep 7: Train=0.5059 | Val=0.5075
   Ep 8: Train=0.4940 | Val=0.5077
   Ep 9: Train=0.5029 | Val=0.4622
   Ep 10: Train=0.5111 | Val=0.4778
       Fold Score: 0.46217
   >>> Fold 2/5...
   Ep 1: Train=0.5109 | Val=0.5021
   Ep 2: Train=0.4993 | Val=0.4780
   Ep 3: Train=0.5023 | Val=0.4935
   Ep 4: Train=0.5060 | Val=0.5227
   Ep 5: Train=0.5090 | Val=0.4985
   Ep 6: Train=0.5104 | Val=0.5135
   Ep 7: Train=0.4987 | Val=0.4935
   Ep 8: Train=0.5067 | Val=0.5126
   Ep 9: Train=0.4960 | Val=0.4960
   Ep 10: Train=0.4961 | Val=0.5053
       Fold Score: 0.47802
   >>> Fold 3/5...
   Ep 1: Train=0.5047 | Val=0.4951
   Ep 2: Train=0.4974 | Val=0.4979
   Ep 3: Train=0.5192 | Val=0.5352
   Ep 4: Train=0.5098 | Val=0.5075
   Ep 5: Train=0.5034 | Val=0.5110
   Ep 6: Train=0.5097 | Val=0.4883
   Ep 7: Train=0.5014 | Val=0.4865
   Ep 8: Train=0.4898 | Val=0.5308
   Ep 9: Train=0.5112 | Val=0.4855
   Ep 10: Train=0.5004 | Val=0.5346
       Fold Score: 0.48549
   >>> Fold 4/5...
   Ep 1: Train=0.5176 | Val=0.5332
   Ep 2: Train=0.5017 | Val=0.4971
   Ep 3: Train=0.5005 | Val=0.4996
   Ep 4: Train=0.5014 | Val=0.5048
   Ep 5: Train=0.5078 | Val=0.5145
   Ep 6: Train=0.5057 | Val=0.5137
   Ep 7: Train=0.4972 | Val=0.4829
   Ep 8: Train=0.5149 | Val=0.5105
   Ep 9: Train=0.4945 | Val=0.4974
   Ep 10: Train=0.5011 | Val=0.5229
       Fold Score: 0.48293
   >>> Fold 5/5...
   Ep 1: Train=0.5104 | Val=0.4963
   Ep 2: Train=0.5065 | Val=0.5035
   Ep 3: Train=0.5087 | Val=0.5029
   Ep 4: Train=0.5026 | Val=0.5326
   Ep 5: Train=0.5031 | Val=0.5143
   Ep 6: Train=0.5070 | Val=0.5132
   Ep 7: Train=0.5029 | Val=0.5095
   Ep 8: Train=0.5041 | Val=0.5053
   Ep 9: Train=0.5001 | Val=0.5065
   Ep 10: Train=0.4915 | Val=0.5051
       Fold Score: 0.49627

FINAL SUMMARY:
Model A (MSE): 1.72921
Model B (Huber): 0.48097
