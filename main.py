import pandas as pd
import numpy as np
import ydata_profiling
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import argsort
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, KFold, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix, f1_score, \
    precision_score, mean_squared_error, mean_absolute_error, r2_score
from ydata_profiling import ProfileReport

df = pd.read_csv(r"C:\\Users\Dell\Downloads\concrete.csv")
print(df.head().to_string())

columns = ["cement", "slag", "ash", "water", "superplasticizer", "coarseaggregate", "fineaggregate", "age", "strength"]
df.columns = columns
print(df.head().to_string())
print((df.corr().to_string()))
ydata_profiling.ProfileReport(df)
# ProfileReport(df).to_file(output_file="Concrete report.html")
duplicates = df[df.duplicated(keep=False)]
print(duplicates.to_string())

X = df.drop("strength", axis=1)
y = df["strength"]
feat = X.columns
plt.style.use("ggplot")
plt.figure(figsize=(10, 10))
sns.boxplot(data=X)
plt.xticks(ticks=np.arange(len(feat)), labels=feat)
# plt.show()
scl = MinMaxScaler()
scl_data = scl.fit_transform(df)
print(scl_data.shape)
X = scl_data[:, :8]
y = scl_data[:, 8]

plt.style.use("ggplot")
plt.figure(figsize=(10, 20))
sns.boxplot(data=X)
plt.xticks(ticks=np.arange(len(feat)), labels=feat)


# plt.show()


def format_text(size, colour, weight="heavy"):
    return {"size": size, "color": colour, "weight": weight}


plt.style.use("bmh")
fig = plt.figure(figsize=(10, 10), constrained_layout=True)
gs = gridspec.GridSpec(4, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
df.cement.plot.kde()
ax1.set_title("Cement", **format_text(20, "#7400b8"))
ax2 = fig.add_subplot(gs[0, 1])
df.slag.plot.kde()
ax2.set_title("Slag", **format_text(20, "#7400b8"))
ax3 = fig.add_subplot(gs[1, 0])
df.ash.plot.kde()
ax3.set_title("Ash", **format_text(20, "#7400b8"))
ax4 = fig.add_subplot(gs[1, 1])
df.water.plot.kde()
ax4.set_title("Water", **format_text(20, "#7400b8"))
ax5 = fig.add_subplot(gs[2, 0])
df.superplasticizer.plot.kde()
ax5.set_title("Superplasticizer", **format_text(20, "#7400b9"))
ax6 = fig.add_subplot(gs[2, 1])
df.coarseaggregate.plot.kde()
ax6.set_title("Coarseaggregate", **format_text(20, "#7400b9"))
ax7 = fig.add_subplot(gs[3, 0])
df.fineaggregate.plot.kde()
ax7.set_title("Fineaggregate", **format_text(20, "#7400b9"))
ax8 = fig.add_subplot(gs[3, 1])
df.age.plot.kde()
ax8.set_title("Age", **format_text(20, "#7400b9"))
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lin_model = LinearRegression()
score = cross_val_score(lin_model, X_train, y_train, cv=5)
print("score", score)
print("accuracy %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

k_fold = KFold(n_splits=5)
score = cross_val_score(lin_model, X_train, y_train, cv=k_fold, scoring="neg_mean_squared_error")
print("score", score)
print("accuracy %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

loo = LeaveOneOut()
loo.get_n_splits(X)
scores = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    model = LinearRegression()
    model.fit(X_train, y_train)
    scores.append(mean_absolute_error(y_test, model.predict(X_test)))

print("average scores :%0.2f" % np.mean(scores))

lasso = Lasso(alpha=1e-5)
lasso.fit(X, y)
print("coefficients: %(coef)s, incercept: %(intercept)s" % {"coef": lasso.coef_, "intercept": lasso.intercept_})
print("mean_absolute_error:", mean_absolute_error(y, lasso.predict(X)))
print("mean_squared_error:", mean_squared_error(y, lasso.predict(X)))
print("r2_score:", r2_score(y, lasso.predict(X)))

ridge = Ridge(alpha=1e-3)
ridge.fit(X, y)
print("coefficients: %(coef)s, incercept: %(intercept)s" % {"coef": ridge.coef_, "intercept": ridge.intercept_})
print("mean_absolute_error:", mean_absolute_error(y, ridge.predict(X)))
print("mean_squared_error:", mean_squared_error(y, ridge.predict(X)))
print("r2_score:", r2_score(y, ridge.predict(X)))


def eval_data(X, y, model_name, model):
    ev1 = {"model": [], "R2_score": [], "MAE": [], "MSE": []}
    ev1["model"].append(model_name)
    ev1["R2_score"].append(r2_score(y, model.predict(X)))
    ev1["MAE"].append(mean_absolute_error(y, model.predict(X)))
    ev1["MSE"].append(mean_squared_error(y, model.predict(X)))
    return ev1


rev1 = eval_data(X, y, "ridge model", ridge)
lev1 = eval_data(X, y, "lasso model", lasso)
print(rev1, lev1)

rfr = RandomForestRegressor()
rfr.fit(X, y)
print("mean_absolute_error:", mean_absolute_error(y, rfr.predict(X)))
print("mean_squared_error RFR:", mean_squared_error(y, rfr.predict(X)))
print("r2_score:", r2_score(y, rfr.predict(X)))

lr = LinearRegression()
lr.fit(X, y)
print("mean_absolute_error:", mean_absolute_error(y, lr.predict(X)))
print("mean_squared_error:", mean_squared_error(y, lr.predict(X)))
print("r2_score:", r2_score(y, lr.predict(X)))


def append_data(*args):
    ev1 = {}
    for data in args:
        for key in data.keys():
            if key in ev1:
                ev1[key].extend(data[key])
            else:
                ev1[key] = data[key]
    return ev1


ev1 = append_data(rev1, lev1)
ev1 = append_data(eval_data(X, y, "Random Forest Regressor", rfr), ev1)
ev1 = append_data(eval_data(X, y, "Linear Regression", lr), ev1)

print("yes", ev1)


def plot_regress(df):
    sns.set_palette(sns.color_palette("rocket"))
    super_title = {"size": 18, "color": "#c5283d", "weight": "extra bold"}
    sub_title = {"size": 14, "color": "#e06777", "weight": "bold"}
    colors = np.array([[156, 137, 184], [239, 195, 230], [184, 190, 120], [231, 115, 117]])
    colors = colors / 255
    df = pd.DataFrame(df)
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(df.model, df.R2_score, color=colors[0])
    ax1.set_xlim(0, 1)
    ax1.set_title("R2_score", **sub_title)
    ax1.tick_params(labelbottom=True, labelleft=True)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(df.model, df.MAE, color=colors[1])
    ax2.set_xlim(0, 0.5)
    ax2.set_title("MAE", **sub_title)
    ax2.tick_params(labelbottom=True, labelleft=True)
    ax3 = fig.add_subplot(gs[1,:])
    ax3.barh(df.model, df.MSE, color=colors[2])
    ax3.set_xlim(0, 0.2)
    ax3.set_title("MSE", **sub_title)
    ax3.tick_params(labelbottom=True, labelleft=True)
    fig.suptitle("Evaluation", **super_title)
    ax4.tick_params(labelbottom=True, labelleft=True)

    plt.show()

plot_regress(ev1)