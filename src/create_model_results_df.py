import pandas as pd

def create_results_df(y_test, pred, selected_ticker, tickers):
    # Create separate DataFrames for y_test and lr_pred for the selected ticker
    y_test_df = pd.DataFrame({'Atual': y_test[selected_ticker]}, index=y_test.index)
    lr_pred_df = pd.DataFrame({'Predicted': pred[:, tickers.to_list().index(selected_ticker)]}, index=y_test.index)

    # Combine them into a single DataFrame
    results_df = pd.concat([y_test_df, lr_pred_df], axis=1)

    return results_df