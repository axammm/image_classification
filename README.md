# Image classification
There are several types of common concrete cracks namely hairline cracks which usually develop in concrete foundation as the concrete cures, shrinkage cracks which occur while the concrete is curing, settlement cracks which happen when part of concrete sinks or when the ground underneath the slab isn’t compacted properly as well as structural cracks which form due to incorrect design. Concrete cracks may endanger the safety and durability of a building if not being identified quickly and left untreated. So, the purpose of the project is to perform image classification to classify concretes with or without cracks. The developed model is impactful and may save thousands of lives.
 
## Project Description
 Steps involved in this project included:
 1. Data loading 
 - The datase was loaded using special method as tensorflow dataset.

 2. Split
 - By using the cardinality method, all the data can be classified into 3 category, train set, test set and validation set. It separated batches by batches.

 3. Convert the the Batch dataset into Prefetch Dataset
 - Afterwards,the batch types of dataset converted into prefetch dataset using Autotune.

 4. Create a 'model' for augmentation.

 5. Apply the transfer learning method
 
 

 ### Acknowledgement 
 - Special thanks to the provider of the dataset.
 1. Source of dataset : https://data.mendeley.com/datasets/5y9wdsg2zt/2
