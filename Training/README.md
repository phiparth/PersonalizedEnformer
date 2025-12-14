The final models A & B were trained using complete pre processed Geuvadis Dataset. We used Huber loss for comparison among these 2 models and as expected model B performed better when trained on the complete dataset.

The final output of the script provided is as follows:

>>> Loading Data Manager...
   Samples: 455
   Caching Expression...
   Parsing Genes & Calculating Stats...
   Valid Genes: 5335

========================================
[MODEL A] Original | Fixed 80/20 Split
========================================
   Ep 1: Train=605.7739 | Val=369.1155
   Ep 2: Train=677.7202 | Val=544.2952
   Ep 3: Train=400.6927 | Val=415.1619
   Ep 4: Train=365.5516 | Val=373.8269
   Ep 5: Train=495.2819 | Val=603.7257
   Ep 6: Train=558.8458 | Val=118.3083
   Ep 7: Train=699.0281 | Val=330.4861
   Ep 8: Train=403.7441 | Val=339.8810
   Ep 9: Train=400.9455 | Val=419.4319
   Ep 10: Train=658.3527 | Val=1061.5912
Model A Score: 118.30830

========================================
[MODEL B] SwiGLU | 5-Fold Cross Validation
========================================
   >>> Running Fold 1/5...
   Ep 1: Train=349.3034 | Val=1303.1471
   Ep 2: Train=483.7309 | Val=127.5899
   Ep 3: Train=529.0118 | Val=350.9505
   Ep 4: Train=374.6188 | Val=744.5565
   Ep 5: Train=481.0391 | Val=308.6046
   Ep 6: Train=622.2447 | Val=1156.0350
   Ep 7: Train=455.7476 | Val=235.8017
   Ep 8: Train=455.6249 | Val=1179.0845
   Ep 9: Train=370.5511 | Val=507.7365
   Ep 10: Train=331.0943 | Val=1011.8687
       Fold 1 Score: 127.58992
   >>> Running Fold 2/5...
   Ep 1: Train=610.1194 | Val=713.1568
Model A Score: 51.75326

========================================
[MODEL B] SwiGLU | 5-Fold Cross Validation
========================================
   >>> Running Fold 1/5...
       Fold 1 Score: 26.78385
   >>> Running Fold 2/5...
       Fold 2 Score: 0.50178
   >>> Running Fold 3/5...
       Fold 3 Score: 34.68949
   >>> Running Fold 4/5...
       Fold 4 Score: 0.45485
   >>> Running Fold 5/5...
       Fold 5 Score: 0.50647

########################################
 FINAL RESULTS (Huber Loss)
########################################
Model A (Original, Fixed): 51.75326
Model B (SwiGLU, 5-Fold):  12.58729
WINNER: Model B (SwiGLU)
