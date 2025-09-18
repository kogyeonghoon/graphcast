# TIGGE ENS Ensemble Download Script

This script downloads TIGGE (THORPEX Interactive Grand Global Ensemble) ENS ensemble data from ECMWF for weather prediction evaluation.

## Overview

The script downloads 15-day forecasts at 12-hour steps starting from 6:00 and 18:00 UTC every day in 2019, with the following specifications:

- **Forecast length**: 15 days (360 hours)
- **Time steps**: 12-hour intervals (0, 12, 24, 36, ..., 360 hours)
- **Initialization times**: 06:00 and 18:00 UTC daily
- **Variables**: 
  - `t2m`: 2-meter temperature
  - `tp`: Total precipitation  
  - `gh`: Geopotential height at 500hPa
- **Forecast types**:
  - `cf`: Control forecast (1 member)
  - `pf`: Perturbed forecast (50 ensemble members)
- **Resolution**: 0.5° x 0.5° global grid

## Prerequisites

### 1. ECMWF API Access

You need to register for ECMWF API access:

1. Visit the [ECMWF TIGGE portal](http://apps.ecmwf.int/datasets/data/tigge)
2. Click "login" to complete user registration
3. After logging in, retrieve your API key at https://api.ecmwf.int/v1/key/
4. Create a `.ecmwfapirc` file in your home directory:

```bash
nano ~/.ecmwfapirc
```

Add the following content:
```json
{
  "url"   : "https://api.ecmwf.int/v1",
  "key"   : "YOUR_API_KEY_HERE",
  "email" : "your.email@example.com"
}
```

### 2. Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements_tigge.txt
```

Or use the setup script:

```bash
./setup_tigge_download.sh
```

## Usage

### Basic Usage

Download all of 2019:
```bash
python tigge_ens_download.py
```

### Advanced Usage

Download specific months:
```bash
python tigge_ens_download.py --start_month 6 --end_month 8
```

Set custom output directory:
```bash
python tigge_ens_download.py --base_dir /path/to/tigge_data
```

Download from a specific day:
```bash
python tigge_ens_download.py --start_day 15
```

### Command Line Arguments

- `--year`: Year to download (default: 2019)
- `--base_dir`: Base directory for storing data (default: ./tigge_data)
- `--start_month`: Starting month (default: 1)
- `--end_month`: Ending month (default: 12)
- `--start_day`: Starting day of the month (default: 1)
- `--check_config`: Check ECMWF API configuration and exit

## Output Structure

The data is organized in the following directory structure:

```
base_dir/
└── TIGGE/
    └── data/
        └── ECMWF/
            ├── t2m/                    # 2-meter temperature
            │   ├── cf/                 # Control forecast
            │   │   └── 2019/
            │   │       ├── 01/
            │   │       │   ├── 06/    # 06:00 UTC
            │   │       │   └── 18/    # 18:00 UTC
            │   │       └── ...
            │   └── pf/                 # Perturbed forecast (50 members)
            │       └── 2019/
            │           └── ...
            ├── tp/                     # Total precipitation
            │   └── ...
            └── gh/                     # Geopotential height at 500hPa
                └── ...
```

### File Naming Convention

Files are named as:
```
{variable}_6hr_ECMWF_{type}_GLO-05_{YYYYMMDD}.grib
```

Examples:
- `t2m_6hr_ECMWF_cf_GLO-05_20190101.grib` - Control forecast for 2m temperature on Jan 1, 2019
- `tp_6hr_ECMWF_pf_GLO-05_20190101.grib` - Perturbed forecast for precipitation on Jan 1, 2019

## Data Details

### Variables

| Variable | ECMWF Code | Description | Level Type |
|----------|------------|-------------|------------|
| `t2m` | 167 | 2-meter temperature | Surface |
| `tp` | 228228 | Total precipitation | Surface |
| `gh` | 156 | Geopotential height | 500hPa |

### Forecast Steps

The script downloads forecasts at the following lead times:
- 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 324, 336, 348, 360 hours

### Ensemble Members

- **Control forecast (cf)**: 1 member
- **Perturbed forecast (pf)**: 50 ensemble members (numbered 1-50)

## Notes

- The script includes error handling and will continue downloading even if some requests fail
- A small delay (0.5 seconds) is added between requests to avoid overwhelming the server
- Files that already exist will be skipped
- The script provides progress updates and a summary at the end
- ECMWF has rate limits, so downloads may take several hours for a full year

## Troubleshooting

### Common Issues

1. **API Configuration Error**
   ```
   Error: ECMWF API configuration file not found.
   ```
   Solution: Follow the setup instructions above to create the `.ecmwfapirc` file.

2. **Import Error**
   ```
   Error: ecmwfapi package not found.
   ```
   Solution: Install the required package: `pip install ecmwf-api-client`

3. **Request Failures**
   - Check your internet connection
   - Verify your API key is valid
   - ECMWF may have temporary service issues

### Getting Help

- Check the [ECMWF API documentation](https://confluence.ecmwf.int/display/WEBAPI/Accessing+ECMWF+Public+Datasets)
- Visit the [TIGGE portal](http://apps.ecmwf.int/datasets/data/tigge) for data information
- Check your request status at https://apps.ecmwf.int/webmars/joblist/

## License

This script is based on the [AusClimateService/TIGGE](https://github.com/AusClimateService/TIGGE) repository and follows the same licensing terms. 