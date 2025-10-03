import os
import time
import logging
import numpy as np
import pandas as pd 
import requests # To make requests through API's
from dotenv import load_dotenv # load's data from environment variables
from sqlalchemy import create_engine, text #Connect to Databases
from requests.adapters import HTTPAdapter 
from urllib3.util.retry import Retry

# ------------------- Logging -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('etl_pipeline.log', encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------- EVM Chains (Etherscan v2 supports chainid param) -------------------
EVM_CHAINS = {
    "Ethereum Mainnet": 1,
    "BNB Smart Chain Mainnet": 56,
    "Polygon Mainnet": 137,
    "Arbitrum One Mainnet": 42161,
    "Arbitrum Nova Mainnet": 42170,
     # If unsupported, it will just return empty
}

# ------------------- Transaction Types (config-driven & extendable) -------------------
# You can add "nft_transfers": {"endpoint": "tokennfttx", ...} or "erc1155_transfers": {"endpoint": "token1155tx", ...}
TX_TYPES = {
    "normal": {
        "endpoint": "txlist",
        "table": "transactions",
        "pk": ["hash", "tx_type", "chain_id"]
    },
    "internal": {
        "endpoint": "txlistinternal",
        "table": "transactions",
        "pk": ["hash", "tx_type", "chain_id"]
    },
    "token_transfers": {
        "endpoint": "tokentx",
        "table": "token_transfers",
        "pk": ["tx_hash", "log_index", "chain_id"]
    },
}

# ------------------- Load Config -------------------
def load_config():
    load_dotenv()
    api_key = os.getenv("ETHERSCAN_API_KEY")
    wallet = os.getenv("WALLET_ADDRESS")
    db_url = os.getenv("DB_URL")
    if not all([api_key, wallet, db_url]):
        raise ValueError("Missing required environment variables (ETHERSCAN_API_KEY, WALLET_ADDRESS, DB_URL)")
    logger.info("Configuration loaded successfully")
    return api_key, wallet, db_url

# ------------------- Database Setup -------------------
def setup_database(db_url: str):
    engine = create_engine(db_url)
    schema_sql = """
    CREATE TABLE IF NOT EXISTS transactions (
        hash TEXT NOT NULL,
        tx_type TEXT NOT NULL,             -- 'normal' or 'internal'
        chain_id INT NOT NULL,
        network TEXT,
        block_number BIGINT,
        time_stamp TIMESTAMP,
        from_address TEXT,
        to_address TEXT,
        value NUMERIC,
        gas NUMERIC,
        gas_price NUMERIC,
        tx_fee NUMERIC,
        tx_status TEXT,
        PRIMARY KEY (hash, tx_type, chain_id)
    );

    CREATE TABLE IF NOT EXISTS token_transfers (
        tx_hash TEXT NOT NULL,
        log_index INT NOT NULL,
        chain_id INT NOT NULL,
        network TEXT,
        time_stamp TIMESTAMP,
        contract_address TEXT,
        token_symbol TEXT,
        token_decimals INT,
        from_address TEXT,
        to_address TEXT,
        value NUMERIC,
        gas NUMERIC,
        gas_price NUMERIC,
        tx_fee NUMERIC,
        PRIMARY KEY (tx_hash, log_index, chain_id)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(schema_sql))
    logger.info("Database tables created/verified successfully")
    return engine

# ------------------- HTTP Client (Etherscan v2) -------------------
class EtherscanClient:
    def __init__(self, api_key: str, chain_id: int):
        self.api_key = api_key
        self.chain_id = chain_id
        self.base_url = "https://api.etherscan.io/v2/api"
        self.session = self._setup_session()

    def _setup_session(self):
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def fetch_with_pagination(self, endpoint: str, address: str) -> pd.DataFrame:
        all_tx = []
        start_block = 0
        max_pages = 100
        page = 1

        while page <= max_pages:
            url = (
                f"{self.base_url}?chainid={self.chain_id}&module=account&action={endpoint}"
                f"&address={address}&startblock={start_block}&endblock=99999999&sort=asc&apikey={self.api_key}"
            )
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") != "1" or not data.get("result"):
                    break
                chunk = data.get("result", [])
                all_tx.extend(chunk)

                # If we got less than 10k in one go, likely done for this range
                if len(chunk) < 10000:
                    break

                # Advance the window by last block + 1
                last_block = int(chunk[-1].get("blockNumber", "0"))
                start_block = last_block + 1
                page += 1
                time.sleep(0.2)  # gentle rate limit
            except Exception as e:
                logger.error(f"Fetch error for {endpoint} (chain {self.chain_id}): {e}")
                break

        return pd.DataFrame(all_tx) if all_tx else pd.DataFrame()

# ------------------- Helpers -------------------
def get_series(df: pd.DataFrame, col: str, default=None):
    """Return column if exists, else a Series filled with default (aligned to df)."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)

def to_numeric(s: pd.Series, divide_by=None, default=0):
    out = pd.to_numeric(s, errors='coerce')
    out = out.fillna(default)
    if divide_by is not None:
        out = out.astype(float) / divide_by
    return out

def to_datetime_from_unix(s: pd.Series):
    return pd.to_datetime(pd.to_numeric(s, errors='coerce'), unit='s', errors='coerce')

# ------------------- Cleaners -------------------
def clean_normal(df: pd.DataFrame, chain_id: int, network: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "hash","tx_type","chain_id","network","block_number","time_stamp",
            "from_address","to_address","value","gas","gas_price","tx_fee","tx_status"
        ])

    out = pd.DataFrame(index=df.index)
    out["hash"] = get_series(df, "hash")
    out["tx_type"] = "normal"
    out["chain_id"] = chain_id
    out["network"] = network
    out["block_number"] = pd.to_numeric(get_series(df, "blockNumber"), errors='coerce')
    out["time_stamp"] = to_datetime_from_unix(get_series(df, "timeStamp"))
    out["from_address"] = get_series(df, "from")
    out["to_address"] = get_series(df, "to")
    out["value"] = to_numeric(get_series(df, "value"), divide_by=1e18, default=0)

    gas = to_numeric(get_series(df, "gas"), default=0)
    gas_price = to_numeric(get_series(df, "gasPrice"), divide_by=1e9, default=0)  # Gwei
    out["gas"] = gas
    out["gas_price"] = gas_price
    out["tx_fee"] = gas * gas_price  # in (gas units * gwei) -> gwei; keep as is or convert to native if you prefer

    # tx_status can be 'txreceipt_status' (Ethereum only) or 'isError' (0/1)
    txr = get_series(df, "txreceipt_status", default=None)
    is_err = get_series(df, "isError", default=None)
    # Prefer txreceipt_status; else derive from isError; else None
    out["tx_status"] = np.where(
        txr.notna(), txr,
        np.where(is_err.notna(), is_err, None)
    )

    out = out.dropna(subset=["hash"])
    return out

def clean_internal(df: pd.DataFrame, chain_id: int, network: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "hash","tx_type","chain_id","network","block_number","time_stamp",
            "from_address","to_address","value","gas","gas_price","tx_fee","tx_status"
        ])

    out = pd.DataFrame(index=df.index)
    out["hash"] = get_series(df, "hash")  # parent tx hash
    out["tx_type"] = "internal"
    out["chain_id"] = chain_id
    out["network"] = network
    out["block_number"] = pd.to_numeric(get_series(df, "blockNumber"), errors='coerce')
    out["time_stamp"] = to_datetime_from_unix(get_series(df, "timeStamp"))
    out["from_address"] = get_series(df, "from")
    out["to_address"] = get_series(df, "to")
    out["value"] = to_numeric(get_series(df, "value"), divide_by=1e18, default=0)

    gas = to_numeric(get_series(df, "gas"), default=0)
    # Internal transfers usually don't include gasPrice; default to 0
    gas_price = to_numeric(get_series(df, "gasPrice"), divide_by=1e9, default=0)
    out["gas"] = gas
    out["gas_price"] = gas_price
    out["tx_fee"] = gas * gas_price

    # Derive status from isError if present
    out["tx_status"] = get_series(df, "isError", default=None)

    out = out.dropna(subset=["hash"])
    return out

def clean_token_transfers(df: pd.DataFrame, chain_id: int, network: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "tx_hash","log_index","chain_id","network","time_stamp","contract_address",
            "token_symbol","token_decimals","from_address","to_address","value","gas","gas_price","tx_fee"
        ])

    out = pd.DataFrame(index=df.index)
    # tx hash & log index
    out["tx_hash"] = get_series(df, "hash")
    out["log_index"] = pd.to_numeric(get_series(df, "logIndex", default=0), errors='coerce').fillna(0).astype(int)
    out["chain_id"] = chain_id
    out["network"] = network
    out["time_stamp"] = to_datetime_from_unix(get_series(df, "timeStamp"))
    out["contract_address"] = get_series(df, "contractAddress")
    out["token_symbol"] = get_series(df, "tokenSymbol")
    token_dec = pd.to_numeric(get_series(df, "tokenDecimal", default=18), errors='coerce').fillna(18).astype(int)
    out["token_decimals"] = token_dec
    out["from_address"] = get_series(df, "from")
    out["to_address"] = get_series(df, "to")

    raw_val = pd.to_numeric(get_series(df, "value", default=0), errors='coerce').fillna(0)
    scale = np.power(10.0, token_dec)
    out["value"] = raw_val.astype(float) / scale

    # Token transfer endpoints typically don't return gas/gasPrice; default to 0
    gas = to_numeric(get_series(df, "gas", default=0), default=0)
    gas_price = to_numeric(get_series(df, "gasPrice", default=0), divide_by=1e9, default=0)
    out["gas"] = gas
    out["gas_price"] = gas_price
    out["tx_fee"] = gas * gas_price

    out = out.dropna(subset=["tx_hash"])
    return out

# ------------------- Spam Token Filtering -------------------
def filter_spam_tokens(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[df["value"] > 0]
    # crude symbol filter
    df = df[~df["token_symbol"].str.contains(r"[\s\|/]|0x", na=False)]
    return df

# ------------------- Safe Insert (de-dup) -------------------
def insert_safe(df: pd.DataFrame, table_name: str, conn, pk_cols: list) -> int:
    if df.empty:
        return 0
    pk_list = ",".join(pk_cols)
    existing = pd.read_sql(f"SELECT {pk_list} FROM {table_name}", conn)
    if not existing.empty:
        merged = df.merge(existing, on=pk_cols, how="left", indicator=True)
        df_to_insert = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    else:
        df_to_insert = df.copy()

    if not df_to_insert.empty:
        df_to_insert.to_sql(table_name, conn, if_exists="append", index=False, method="multi")
        return len(df_to_insert)
    return 0

# ------------------- Main ETL -------------------
def run_etl():
    api_key, wallet, db_url = load_config()
    engine = setup_database(db_url)

    for network, chain_id in EVM_CHAINS.items():
        logger.info(f"Fetching data for {network} (ID: {chain_id})")
        client = EtherscanClient(api_key, chain_id)

        # NORMAL
        normal_raw = client.fetch_with_pagination(TX_TYPES["normal"]["endpoint"], wallet)
        normal_clean = clean_normal(normal_raw, chain_id, network)

        # INTERNAL
        internal_raw = client.fetch_with_pagination(TX_TYPES["internal"]["endpoint"], wallet)
        internal_clean = clean_internal(internal_raw, chain_id, network)

        # TOKEN TRANSFERS
        token_raw = client.fetch_with_pagination(TX_TYPES["token_transfers"]["endpoint"], wallet)
        token_clean = clean_token_transfers(token_raw, chain_id, network)
        token_clean = filter_spam_tokens(token_clean)

        with engine.begin() as conn:
            n_inserted = insert_safe(
                normal_clean, TX_TYPES["normal"]["table"], conn, TX_TYPES["normal"]["pk"]
            )
            i_inserted = insert_safe(
                internal_clean, TX_TYPES["internal"]["table"], conn, TX_TYPES["internal"]["pk"]
            )
            t_inserted = insert_safe(
                token_clean, TX_TYPES["token_transfers"]["table"], conn, TX_TYPES["token_transfers"]["pk"]
            )

        logger.info(f"{network}: Inserted {n_inserted} normal, {i_inserted} internal, {t_inserted} token transfers")

if __name__ == "__main__":
    try:
        run_etl()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
