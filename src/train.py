from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATA_DIR = Path(r"E:\project\house_price")


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):

    # 房屋类型编号， 出售月份， 出售年份作为类别变量处理， 其余作为数值类型处理
    categorical_as_text = ["MSSubClass", "MoSold", "YrSold"]      
    numeric_columns = [
        "LotFrontage",
        "LotArea",
        "OverallQual",
        "OverallCond",
        "YearBuilt",
        "YearRemodAdd",
        "MasVnrArea",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "LowQualFinSF",
        "GrLivArea",
        "BsmtFullBath",
        "BsmtHalfBath",
        "FullBath",
        "HalfBath",
        "BedroomAbvGr",
        "KitchenAbvGr",
        "TotRmsAbvGrd",
        "Fireplaces",
        "GarageYrBlt",
        "GarageCars",
        "GarageArea",
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
        "PoolArea",
        "MiscVal",
    ]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HouseFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for column in self.numeric_columns:
            if column in X.columns:
                X[column] = pd.to_numeric(X[column], errors="coerce")

        for column in self.categorical_as_text:
            if column in X.columns:
                X[column] = X[column].astype("string")

        numeric_defaults = {
            "TotalBsmtSF": 0,
            "1stFlrSF": 0,
            "2ndFlrSF": 0,
            "LowQualFinSF": 0,
            "BsmtFullBath": 0,
            "BsmtHalfBath": 0,
            "FullBath": 0,
            "HalfBath": 0,
            "YearBuilt": np.nan,
            "YearRemodAdd": np.nan,
            "YrSold": np.nan,
            "GarageArea": 0,
            "GarageCars": 0,
            "Fireplaces": 0,
            "PoolArea": 0,
            "WoodDeckSF": 0,
            "OpenPorchSF": 0,
            "EnclosedPorch": 0,
            "3SsnPorch": 0,
            "ScreenPorch": 0,
        }
        values = {
            column: pd.to_numeric(X[column], errors="coerce").fillna(default)
            for column, default in numeric_defaults.items()
            if column in X.columns
        }

        #总面积 = 地下室面积 + 一楼面积 + 二楼面积
        if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(values):
            X["TotalSF"] = values["TotalBsmtSF"] + values["1stFlrSF"] + values["2ndFlrSF"]
        
        #总浴室数量 = 完整浴室 + 0.5*半浴室 + 地下室完整浴室 + 0.5*地下室半浴室
        if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(values):
            X["TotalBath"] = (
                values["FullBath"]
                + 0.5 * values["HalfBath"]
                + values["BsmtFullBath"]
                + 0.5 * values["BsmtHalfBath"]
            )

        #出售时房龄 = 出售年份 - 建造年份
        if {"YrSold", "YearBuilt"}.issubset(values):
            X["HouseAgeAtSale"] = values["YrSold"] - values["YearBuilt"]
        #翻新时间 = 出售年份 - 翻新年份
        if {"YrSold", "YearRemodAdd"}.issubset(values):
            X["YearsSinceRemodel"] = values["YrSold"] - values["YearRemodAdd"]
        #房子是否翻新过 = （翻新年份！=建造年份）
        if {"YearBuilt", "YearRemodAdd"}.issubset(values):
            X["WasRemodeled"] = (values["YearRemodAdd"] != values["YearBuilt"]).astype(int)
        #是否有车库
        if {"GarageArea", "GarageCars"}.issubset(values):
            X["HasGarage"] = ((values["GarageArea"] > 0) | (values["GarageCars"] > 0)).astype(int)
        #是否有地下室
        if "TotalBsmtSF" in values:
            X["HasBasement"] = (values["TotalBsmtSF"] > 0).astype(int)
        #是否有壁炉
        if "Fireplaces" in values:
            X["HasFireplace"] = (values["Fireplaces"] > 0).astype(int)
        #是否有泳池
        if "PoolArea" in values:
            X["HasPool"] = (values["PoolArea"] > 0).astype(int)

        #总户外面积
        porch_columns = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
        if all(column in values for column in porch_columns):
            X["TotalPorchSF"] = sum(values[column] for column in porch_columns)

        return X

#用于交叉验证中的表现
def rmsle_from_log_predictions(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    predictions = np.maximum(estimator.predict(X), 0)
    return -root_mean_squared_error(np.log1p(y), np.log1p(predictions))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_pipeline(model: BaseEstimator, preprocessor: ColumnTransformer) -> Pipeline:
    regressor = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return Pipeline(
        steps=[
            ("features", HouseFeatureEngineer()),
            ("preprocess", preprocessor),
            ("model", regressor),
        ]
    )


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv in {data_dir}. "
            "Download the competition data from Kaggle first."
        )

    train = pd.read_csv(train_path, keep_default_na=False, na_values=[""])
    test = pd.read_csv(test_path, keep_default_na=False, na_values=[""])

    y = train["SalePrice"]
    X = train.drop(columns=["Id", "SalePrice"])
    return X, y, test


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, test = load_data(data_dir)
    engineered_X = HouseFeatureEngineer().fit_transform(X)
    preprocessor = build_preprocessor(engineered_X)

    models: dict[str, BaseEstimator] = {
        "ridge": Ridge(alpha=18.0),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=700,
            learning_rate=0.025,
            max_depth=3,
            min_samples_leaf=3,
            subsample=0.75,
            random_state=args.random_state,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.65,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.65,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        ),
    }

    cv = KFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    results = []
    best_name = ""
    best_score = np.inf
    best_pipeline: Pipeline | None = None

    for name, model in models.items():
        pipeline = make_pipeline(model, preprocessor)
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=rmsle_from_log_predictions,
            n_jobs=1,
        )
        mean_rmsle = -scores.mean()
        std_rmsle = scores.std()
        results.append({"model": name, "mean_rmsle": mean_rmsle, "std_rmsle": std_rmsle})
        print(f"{name:18s} RMSLE: {mean_rmsle:.5f} +/- {std_rmsle:.5f}")

        if mean_rmsle < best_score:
            best_score = mean_rmsle
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was trained.")

    results_df = pd.DataFrame(results).sort_values("mean_rmsle")
    results_df.to_csv(output_dir / "cv_results.csv", index=False)

    print(f"\nSelected model: {best_name} (CV RMSLE {best_score:.5f})")
    best_pipeline.fit(X, y)

    predictions = np.maximum(best_pipeline.predict(test), 0)
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": predictions})
    submission.to_csv(output_dir / "submission.csv", index=False)
    joblib.dump(best_pipeline, output_dir / "model.joblib")

    print(f"Wrote {output_dir / 'submission.csv'}")
    print(f"Wrote {output_dir / 'cv_results.csv'}")
    print(f"Wrote {output_dir / 'model.joblib'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model for Kaggle House Prices.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing train.csv and test.csv.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for submission and model artifacts.")
    parser.add_argument("--folds", type=int, default=5, help="Number of K-fold CV splits.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for tree models.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
