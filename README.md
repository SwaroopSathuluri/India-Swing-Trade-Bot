# India Swing Scanner

Static HTML swing-trading reports for Indian equities using open Yahoo Finance data through `yfinance`.

Available universes:

- `india-largecap` for a curated basket of liquid NSE large caps
- `banking` for large private/public bank and NBFC leaders
- `it` for major Indian technology names

## Local use

Generate a local report:

```bat
generate_local_report.cmd india-largecap
generate_local_report.cmd banking
generate_local_report.cmd it
```

The report is written to:

- `india_swing_report.html`
- `index.html`

`index.html` is the GitHub Pages entry file.

## GitHub Pages setup

1. Push this project to your GitHub repository.
2. In `Settings -> Pages`, set the source to `GitHub Actions`.
3. Run the `Publish India Swing Report` workflow once manually.

After that, GitHub Actions will:

- regenerate the report on schedule
- publish the latest `index.html` to GitHub Pages
- let you choose which Indian universe to scan on manual runs

## Notes

- Data comes from Yahoo Finance via `yfinance`, which is convenient for open-data research but is not an exchange-licensed institutional feed.
- The ticker baskets in `universe_lists.py` are curated so you can edit them whenever you want.
- Relative strength is measured versus `^NSEI` by default, or `^NSEBANK` for the banking universe.

## Schedule

The workflow currently refreshes on weekdays at:

- `04:30 UTC`
- `10:30 UTC`
- `15:30 UTC`
