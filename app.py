
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
model.fit(iris.data, iris.target)

st.title("🌸 Iris Flower Predictor")
st.write("Enter the flower measurements below.")

sl = st.number_input("Sepal length (cm)", value=5.1)
sw = st.number_input("Sepal width (cm)", value=3.5)
pl = st.number_input("Petal length (cm)", value=1.4)
pw = st.number_input("Petal width (cm)", value=0.2)

if st.button("Predict"):
    result = model.predict([[sl, sw, pl, pw]])[0]
    st.success(
        "Predicted flower: "
        + iris.target_names[result].title()
    )
