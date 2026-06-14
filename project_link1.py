import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,LogisticRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.metrics import r2_score ,accuracy_score,mean_absolute_error,mean_squared_error,recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline



file=r"C:/Users/justin/Downloads/archive (1)/insurance.csv"
df=pd.read_csv(file)


col_encode=["smoker","region","sex"]

df=pd.get_dummies(data=df,columns=col_encode,drop_first=True)
df[
    ["sex_male","region_northwest","region_southeast","region_southwest","smoker_yes"]
    ]=df[
        ["sex_male","region_northwest","region_southeast","region_southwest","smoker_yes"]
         ].astype(int)
print(df.sample(10))

x=df[["sex_male","region_northwest","region_southeast","region_southwest","bmi","children"]]
y=df["charges"]
y_log=df["smoker_yes"]


x_train,x_test,y_train,y_test,y_log_train,y_log_test = train_test_split(x,y,y_log,train_size=0.8,random_state=12)


# linear regression section
model=LinearRegression()
model.fit(x_train,y_train)
prd = model.predict((x_test))
r2=r2_score(y_test,prd)
mae=mean_absolute_error(y_test,prd)
mse=mean_squared_error(y_test,prd)
rmse=np.sqrt(mse)



# polynomial regression section
best_degree=None
best_poly_score=-999
best_prediction=None
degree=[1,2,3,4,5]
for d in degree:
    poly=make_pipeline(PolynomialFeatures(degree=d),LinearRegression())
    poly.fit(x_train,y_train)
    poly_prd=poly.predict(x_test)
    poly_r2=r2_score(y_test,poly_prd)
    if poly_r2>best_poly_score:
        best_poly_score=poly_r2
        best_degree=d
        best_prediction=poly_prd
poly_mae=mean_absolute_error(y_test,best_prediction)
poly_mse=mean_squared_error(y_test,best_prediction)
poly_rmse=np.sqrt(poly_mse)

# ridge prediction section
parameter=[0.01,0.1,1,2,3,4]
best_ri_prediction = None
best_ri_alpha=None
best_ri_score=-999 
for r in parameter:
  rid=make_pipeline(StandardScaler(),Ridge(alpha=r))
  rid.fit(x_train,y_train)
  ri_prd=rid.predict(x_test)
  ri_r2=r2_score(y_test,ri_prd)
  if ri_r2>best_ri_score:
     best_ri_score=ri_r2
     best_ri_alpha=r
     best_ri_prediction=ri_prd
ri_mae=mean_absolute_error(y_test,best_ri_prediction)
ri_mse=mean_squared_error(y_test,best_ri_prediction)
ri_rmse=np.sqrt(ri_mse)


#lasso prediction section 
parameter=[0.01,0.1,1,2,3,4]
best_las_prediction = None
best_las_alpha=None
best_las_score=-999 
for l in parameter:
  las=make_pipeline(StandardScaler(),Lasso(alpha=l,max_iter=10000))
  las.fit(x_train,y_train)
  las_prd=las.predict(x_test)
  las_r2=r2_score(y_test,las_prd)
  if las_r2>best_las_score:
     best_las_score=las_r2
     best_las_alpha=l
     best_las_prediction=las_prd 
las_mae=mean_absolute_error(y_test,best_las_prediction)
las_mse=mean_squared_error(y_test,best_las_prediction)
las_rmse=np.sqrt(las_mse)

# elastic net prediction section 
parameter=[0.01,0.1,1,2,3,4]
ratio=[0.2,0.5,0.8]
best_el_prediction = None
best_el_alpha=None
best_ratio = None
best_el_score=-999 
for el in parameter:
  for ra in ratio:
    elas=make_pipeline(StandardScaler(),ElasticNet(alpha=el,l1_ratio=ra,max_iter=10000,tol=0.0001))
    elas.fit(x_train,y_train)
    el_prd=elas.predict(x_test)
    el_r2=r2_score(y_test,el_prd)
    if el_r2>best_el_score:
        best_el_score=el_r2
        best_el_alpha=el
        best_ratio=ra
        best_el_prediction=el_prd 
el_mae=mean_absolute_error(y_test,best_el_prediction)
el_mse=mean_squared_error(y_test,best_el_prediction)
el_rmse=np.sqrt(el_mse)


# logistic regression 
log=make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=1000))
log.fit(x_train,y_log_train)
log_prd=log.predict(x_test)
acuracy = accuracy_score(y_log_test,log_prd)
rell=recall_score(y_log_test,log_prd)
prcisin=precision_score(y_log_test,log_prd)
f1=f1_score(y_log_test,log_prd)

print(np.unique(y_log_test,return_counts=True))





# promote all algorithm into logistic to compare with logistic
lin_pro=(prd>33471.97189).astype(int)
poly_pro=(best_prediction>33471.97189).astype(int)
ridge_pro=(best_ri_prediction>33471.97189).astype(int)
las_pro=(best_las_prediction>33471.97189).astype(int)
elas_pro=(best_el_prediction>33471.97189).astype(int)


li_ac=accuracy_score(y_log_test,lin_pro)
po_ac=accuracy_score(y_log_test,poly_pro)
ri_ac=accuracy_score(y_log_test,ridge_pro)
la_ac=accuracy_score(y_log_test,las_pro)
el_ac=accuracy_score(y_log_test,elas_pro)

scores={
   "linear regression":{
      "prediction":round(prd[0],2),
      "mae score":mae ,
      "mse score":mse,
      "rmse score":rmse ,
      "r2 score":r2,
      "accuracy score":li_ac
   },
   "polynomial regression":{
      "prediction":round(best_prediction[0],2),
      "mae score":poly_mae ,
      "mse score":poly_mse ,
      "rmse score":poly_rmse,
      "r2 score":best_poly_score,
      "accuracy score":po_ac
      
   },
   "ridge regression":{
      "prediction":round(best_ri_prediction[0],2),
      "mae score":ri_mae ,
      "mse score":ri_mse ,
      "rmse score":ri_rmse ,
      "r2 score":best_ri_score ,
      "accuracy score":ri_ac
      
   },
   "lasso regression":{
     "prediction":round(best_las_prediction[0],2),
    "mae score":las_mae,
      "mse score":las_mse ,
      "rmse score":las_rmse ,
      "r2 score":best_las_score,
      "accuracy score":la_ac
   },
   "elastic net regression ":{
      "prediction":round(best_el_prediction[0],2),
      "mae score":el_mae,
      "mse score":el_mse,
      "rmse score":el_rmse ,
      "r2 score":best_el_score,
      "accuracy score":el_ac
   }


}



best_model = max(scores,key=lambda x : scores[x]["accuracy score"])
best_score=scores[best_model]
best_acu=scores[best_model]["accuracy score"]
print(f"\nbest parameter we used report for all algorithm:\n")
print(f"Best Polynomial Degree: {best_degree}")
print(f"Best Ridge Alpha: {best_ri_alpha}")
print(f"Best Lasso Alpha: {best_las_alpha}")
print(f"Best ElasticNet Alpha: {best_el_alpha}")
print(f"Best ElasticNet Ratio: {best_ratio}\n")
print(f"\nbest model : {best_model.upper()}:=\n")
for key,value in scores[best_model].items():
   if isinstance(value,(int,float,np.number)):
      print(f"{key:<15} : {round(value,2)}")
   else:
       print(f"{key:<15} : {value}")
print(f"\nLOGISTIC REGRESSION := \n")
print("precision =",round(prcisin,3))
print("accuracy = ",round(acuracy,2))
print("recall = ",rell)
print("f1 score =",round(f1,3))
print(f"\nFINAL COMPARISION REPORT := \n")
diff=best_acu-acuracy
print(f"({best_model}) and (logistic regression) differents in prediction = {round(diff,3)}\n")
