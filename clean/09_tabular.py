# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# hide
from waterfall_chart import plot as waterfall
from treeinterpreter import treeinterpreter
import warnings
from sklearn.inspection import plot_partial_dependence
from IPython.display import Image, display_svg, SVG
from dtreeviz.trees import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from fastai.tabular.all import *
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from kaggle import api
from fastbook import *
import fastbook
import kaggle
fastbook.setup_book()

# +
# hide

pd.options.display.max_rows = 20
pd.options.display.max_columns = 8
# -

# # Tabular Modeling Deep Dive

# ## Categorical Embeddings

# ## Beyond Deep Learning

# ## The Dataset

# ### Kaggle Competitions

path = URLs.path('bluebook')
path

# hide
Path.BASE_PATH = path

# +
if not path.exists():
    path.mkdir()
    kaggle.api.competition_download_cli('bluebook-for-bulldozers', path=path)
    file_extract(path / 'bluebook-for-bulldozers.zip')

path.ls(file_type='text')
# -

# ### Look at the Data

df = pd.read_csv(path / 'TrainAndValid.csv', low_memory=False)

df.columns

df['ProductSize'].unique()

sizes = 'Large', 'Large / Medium', 'Medium', 'Small', 'Mini', 'Compact'

df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True)

dep_var = 'SalePrice'

df[dep_var] = np.log(df[dep_var])

# ## Decision Trees

# ### Handling Dates

df = add_datepart(df, 'saledate')

df_test = pd.read_csv(path / 'Test.csv', low_memory=False)
df_test = add_datepart(df_test, 'saledate')

' '.join(o for o in df.columns if o.startswith('sale'))

# ### Using TabularPandas and TabularProc

procs = [Categorify, FillMissing]

# +
cond = (df.saleYear < 2011) | (df.saleMonth < 10)
train_idx = np.where(cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx), list(valid_idx))
# -

cont, cat = cont_cat_split(df, 1, dep_var=dep_var)

to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)

len(to.train), len(to.valid)

to.show(3)

to1 = TabularPandas(df, procs, ['state', 'ProductGroup', 'Drive_System', 'Enclosure'], [], y_names=dep_var, splits=splits)
to1.show(3)

to.items.head(3)

to1.items[['state', 'ProductGroup', 'Drive_System', 'Enclosure']].head(3)

to.classes['ProductSize']

(path / 'to.pkl').save(to)

# ### Creating the Decision Tree

# hide
to = (path / 'to.pkl').load()

xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y

m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y)

draw_tree(m, xs, size=7, leaves_parallel=True, precision=2)

samp_idx = np.random.permutation(len(y))[:500]
dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
         fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
         orientation='LR')

xs.loc[xs['YearMade'] < 1900, 'YearMade'] = 1950
valid_xs.loc[valid_xs['YearMade'] < 1900, 'YearMade'] = 1950

# +
m = DecisionTreeRegressor(max_leaf_nodes=4).fit(xs, y)

dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
         fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
         orientation='LR')
# -

m = DecisionTreeRegressor()
m.fit(xs, y)


def r_mse(pred, y): return round(math.sqrt(((pred - y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)


m_rmse(m, xs, y)

m_rmse(m, valid_xs, valid_y)

m.get_n_leaves(), len(xs)

m = DecisionTreeRegressor(min_samples_leaf=25)
m.fit(to.train.xs, to.train.y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)

m.get_n_leaves()


# ### Categorical Variables

# ## Random Forests

# +
# hide
# pip install —pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn —U
# -

# ### Creating a Random Forest

def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
                                 max_samples=max_samples, max_features=max_features,
                                 min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)


m = rf(xs, y)

m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)

preds = np.stack([t.predict(valid_xs) for t in m.estimators_])

r_mse(preds.mean(0), valid_y)

plt.plot([r_mse(preds[:i + 1].mean(0), valid_y) for i in range(40)])

# ### Out-of-Bag Error

r_mse(m.oob_prediction_, y)

# ## Model Interpretation

# ### Tree Variance for Prediction Confidence

preds = np.stack([t.predict(valid_xs) for t in m.estimators_])

preds.shape

preds_std = preds.std(0)

preds_std[:5]


# ### Feature Importance

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols': df.columns, 'imp': m.feature_importances_}
                        ).sort_values('imp', ascending=False)


fi = rf_feat_importance(m, xs)
fi[:10]


# +
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12, 7), legend=False)

plot_fi(fi[:30])
# -

# ### Removing Low-Importance Variables

to_keep = fi[fi.imp > 0.005].cols
len(to_keep)

xs_imp = xs[to_keep]
valid_xs_imp = valid_xs[to_keep]

m = rf(xs_imp, y)

m_rmse(m, xs_imp, y), m_rmse(m, valid_xs_imp, valid_y)

len(xs.columns), len(xs_imp.columns)

plot_fi(rf_feat_importance(m, xs_imp))

# ### Removing Redundant Features

cluster_columns(xs_imp)


def get_oob(df):
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=15,
                              max_samples=50000, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(df, y)
    return m.oob_score_


get_oob(xs_imp)

{c: get_oob(xs_imp.drop(c, axis=1)) for c in (
    'saleYear', 'saleElapsed', 'ProductGroupDesc', 'ProductGroup',
    'fiModelDesc', 'fiBaseModel',
    'Hydraulics_Flow', 'Grouser_Tracks', 'Coupler_System')}

to_drop = ['saleYear', 'ProductGroupDesc', 'fiBaseModel', 'Grouser_Tracks']
get_oob(xs_imp.drop(to_drop, axis=1))

xs_final = xs_imp.drop(to_drop, axis=1)
valid_xs_final = valid_xs_imp.drop(to_drop, axis=1)

(path / 'xs_final.pkl').save(xs_final)
(path / 'valid_xs_final.pkl').save(valid_xs_final)

xs_final = (path / 'xs_final.pkl').load()
valid_xs_final = (path / 'valid_xs_final.pkl').load()

m = rf(xs_final, y)
m_rmse(m, xs_final, y), m_rmse(m, valid_xs_final, valid_y)

# ### Partial Dependence

p = valid_xs_final['ProductSize'].value_counts(sort=False).plot.barh()
c = to.classes['ProductSize']
plt.yticks(range(len(c)), c)

ax = valid_xs_final['YearMade'].hist()

# +

fig, ax = plt.subplots(figsize=(12, 4))
plot_partial_dependence(m, valid_xs_final, ['YearMade', 'ProductSize'],
                        grid_resolution=20, ax=ax)
# -

# ### Data Leakage

# ### Tree Interpreter

# hide
warnings.simplefilter('ignore', FutureWarning)


row = valid_xs_final.iloc[:5]

prediction, bias, contributions = treeinterpreter.predict(m, row.values)

prediction[0], bias[0], contributions[0].sum()

waterfall(valid_xs_final.columns, contributions[0], threshold=0.08,
          rotation_value=45, formatting='{:,.3f}')

# ## Extrapolation and Neural Networks

# ### The Extrapolation Problem

# hide
np.random.seed(42)

x_lin = torch.linspace(0, 20, steps=40)
y_lin = x_lin + torch.randn_like(x_lin)
plt.scatter(x_lin, y_lin)

xs_lin = x_lin.unsqueeze(1)
x_lin.shape, xs_lin.shape

x_lin[:, None].shape

m_lin = RandomForestRegressor().fit(xs_lin[:30], y_lin[:30])

plt.scatter(x_lin, y_lin, 20)
plt.scatter(x_lin, m_lin.predict(xs_lin), color='red', alpha=0.5)

# ### Finding Out-of-Domain Data

# +
df_dom = pd.concat([xs_final, valid_xs_final])
is_valid = np.array([0] * len(xs_final) + [1] * len(valid_xs_final))

m = rf(df_dom, is_valid)
rf_feat_importance(m, df_dom)[:6]

# +
m = rf(xs_final, y)
print('orig', m_rmse(m, valid_xs_final, valid_y))

for c in ('SalesID', 'saleElapsed', 'MachineID'):
    m = rf(xs_final.drop(c, axis=1), y)
    print(c, m_rmse(m, valid_xs_final.drop(c, axis=1), valid_y))

# +
time_vars = ['SalesID', 'MachineID']
xs_final_time = xs_final.drop(time_vars, axis=1)
valid_xs_time = valid_xs_final.drop(time_vars, axis=1)

m = rf(xs_final_time, y)
m_rmse(m, valid_xs_time, valid_y)
# -

xs['saleYear'].hist()

filt = xs['saleYear'] > 2004
xs_filt = xs_final_time[filt]
y_filt = y[filt]

m = rf(xs_filt, y_filt)
m_rmse(m, xs_filt, y_filt), m_rmse(m, valid_xs_time, valid_y)

# ### Using a Neural Network

df_nn = pd.read_csv(path / 'TrainAndValid.csv', low_memory=False)
df_nn['ProductSize'] = df_nn['ProductSize'].astype('category')
df_nn['ProductSize'].cat.set_categories(sizes, ordered=True, inplace=True)
df_nn[dep_var] = np.log(df_nn[dep_var])
df_nn = add_datepart(df_nn, 'saledate')

df_nn_final = df_nn[list(xs_final_time.columns) + [dep_var]]

cont_nn, cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var=dep_var)

cont_nn.append('saleElapsed')
cat_nn.remove('saleElapsed')

df_nn_final[cat_nn].nunique()

xs_filt2 = xs_filt.drop('fiModelDescriptor', axis=1)
valid_xs_time2 = valid_xs_time.drop('fiModelDescriptor', axis=1)
m2 = rf(xs_filt2, y_filt)
m_rmse(m, xs_filt2, y_filt), m_rmse(m2, valid_xs_time2, valid_y)

cat_nn.remove('fiModelDescriptor')

procs_nn = [Categorify, FillMissing, Normalize]
to_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn,
                      splits=splits, y_names=dep_var)

dls = to_nn.dataloaders(1024)

y = to_nn.train.y
y.min(), y.max()


learn = tabular_learner(dls, y_range=(8, 12), layers=[500, 250],
                        n_out=1, loss_func=F.mse_loss)

learn.lr_find()

learn.fit_one_cycle(5, 1e-2)

preds, targs = learn.get_preds()
r_mse(preds, targs)

learn.save('nn')

# ### Sidebar: fastai's Tabular Classes

# ### End sidebar

# ## Ensembling

rf_preds = m.predict(valid_xs_time)
ens_preds = (to_np(preds.squeeze()) + rf_preds) / 2

r_mse(ens_preds, valid_y)

# ### Boosting

# ### Combining Embeddings with Other Methods

# ## Conclusion: Our Advice for Tabular Modeling

# ## Questionnaire

# 1. What is a continuous variable?
# 1. What is a categorical variable?
# 1. Provide two of the words that are used for the possible values of a categorical variable.
# 1. What is a "dense layer"?
# 1. How do entity embeddings reduce memory usage and speed up neural networks?
# 1. What kinds of datasets are entity embeddings especially useful for?
# 1. What are the two main families of machine learning algorithms?
# 1. Why do some categorical columns need a special ordering in their classes? How do you do this in Pandas?
# 1. Summarize what a decision tree algorithm does.
# 1. Why is a date different from a regular categorical or continuous variable, and how can you preprocess it to allow it to be used in a model?
# 1. Should you pick a random validation set in the bulldozer competition? If no, what kind of validation set should you pick?
# 1. What is pickle and what is it useful for?
# 1. How are `mse`, `samples`, and `values` calculated in the decision tree drawn in this chapter?
# 1. How do we deal with outliers, before building a decision tree?
# 1. How do we handle categorical variables in a decision tree?
# 1. What is bagging?
# 1. What is the difference between `max_samples` and `max_features` when creating a random forest?
# 1. If you increase `n_estimators` to a very high value, can that lead to overfitting? Why or why not?
# 1. In the section "Creating a Random Forest", just after <<max_features>>, why did `preds.mean(0)` give the same result as our random forest?
# 1. What is "out-of-bag-error"?
# 1. Make a list of reasons why a model's validation set error might be worse than the OOB error. How could you test your hypotheses?
# 1. Explain why random forests are well suited to answering each of the following question:
#    - How confident are we in our predictions using a particular row of data?
#    - For predicting with a particular row of data, what were the most important factors, and how did they influence that prediction?
#    - Which columns are the strongest predictors?
#    - How do predictions vary as we vary these columns?
# 1. What's the purpose of removing unimportant variables?
# 1. What's a good type of plot for showing tree interpreter results?
# 1. What is the "extrapolation problem"?
# 1. How can you tell if your test or validation set is distributed in a different way than your training set?
# 1. Why do we make `saleElapsed` a continuous variable, even although it has less than 9,000 distinct values?
# 1. What is "boosting"?
# 1. How could we use embeddings with a random forest? Would we expect this to help?
# 1. Why might we not always use a neural net for tabular modeling?

# ### Further Research

# 1. Pick a competition on Kaggle with tabular data (current or past) and try to adapt the techniques seen in this chapter to get the best possible results. Compare your results to the private leaderboard.
# 1. Implement the decision tree algorithm in this chapter from scratch yourself, and try it on the datase you used in the first exercise.
# 1. Use the embeddings from the neural net in this chapter in a random forest, and see if you can improve on the random forest results we saw.
# 1. Explain what each line of the source of `TabularModel` does (with the exception of the `BatchNorm1d` and `Dropout` layers).
