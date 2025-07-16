import pandas as pd
import requests
import time
import logging
from typing import Optional, Tuple, Set, Dict
from collections import defaultdict
import signal
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPSGapFiller:
    def __init__(self, rate_limit_delay: float = 1.5):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'GPSGapFiller/1.0'})
        self.interrupted = False
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.info("\n🛑 Interrupt received! Stopping gracefully...")
        self.interrupted = True
        time.sleep(1)
        sys.exit(0)
    
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode with interrupt checking and better error handling."""
        if self.interrupted:
            return None
            
        if pd.isna(address) or not address.strip():
            return None
            
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': address,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 503:
                logger.warning("Service unavailable - API overloaded")
                time.sleep(5)  # Wait longer for overloaded API
                return None
            elif response.status_code == 429:
                logger.warning("Rate limited - waiting longer")
                time.sleep(10)
                return None
                
            response.raise_for_status()
            data = response.json()
            
            if data:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                return (lat, lon)
            else:
                return None
                
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout for: {address}")
            return None
        except Exception as e:
            logger.error(f"Error geocoding '{address}': {str(e)}")
            return None
        finally:
            # Check for interrupt during sleep
            for i in range(int(self.rate_limit_delay * 10)):
                if self.interrupted:
                    return None
                time.sleep(0.1)
    
    def compare_and_analyze_gaps(self, original_file: str, gps_file: str, address_column: str) -> Dict:
        """Compare original and GPS files to identify gaps and statistics."""
        logger.info(f"📊 Analyzing gaps between {original_file} and {gps_file}")
        
        # Load both files
        df_original = pd.read_csv(original_file)
        df_gps = pd.read_csv(gps_file)
        
        logger.info(f"Original file: {len(df_original)} rows")
        logger.info(f"GPS file: {len(df_gps)} rows")
        
        # Check if GPS column exists
        if 'gps' not in df_gps.columns:
            logger.error("GPS column not found in GPS file!")
            return {}
        
        # Find missing GPS entries
        missing_gps = df_gps[df_gps['gps'].isna() | (df_gps['gps'] == '') | (df_gps['gps'].isna())]
        successful_gps = df_gps[df_gps['gps'].notna() & (df_gps['gps'] != '')]
        
        # Get unique addresses for analysis
        missing_addresses = missing_gps[address_column].dropna().unique()
        successful_addresses = successful_gps[address_column].dropna().unique()
        
        # Analyze address patterns for missing ones
        missing_patterns = defaultdict(int)
        for addr in missing_addresses:
            if pd.notna(addr):
                addr_str = str(addr).lower()
                if 'local' in addr_str:
                    missing_patterns['Contains "Local"'] += 1
                if any(char.isdigit() for char in addr_str):
                    missing_patterns['Contains numbers'] += 1
                if ',' in addr_str:
                    missing_patterns['Contains commas'] += 1
                if len(addr_str.split()) > 6:
                    missing_patterns['Long address (>6 words)'] += 1
                if any(word in addr_str for word in ['suite', 'apt', 'unit', '#']):
                    missing_patterns['Contains unit/suite info'] += 1
        
        results = {
            'total_records': len(df_gps),
            'successful_geocodes': len(successful_gps),
            'failed_geocodes': len(missing_gps),
            'success_rate': (len(successful_gps) / len(df_gps)) * 100,
            'unique_missing_addresses': len(missing_addresses),
            'unique_successful_addresses': len(successful_addresses),
            'missing_patterns': dict(missing_patterns),
            'missing_addresses': missing_addresses[:20],  # Sample of first 20
            'missing_df': missing_gps
        }
        
        return results
    
    def try_address_variations(self, address: str) -> Optional[Tuple[float, float]]:
        """Try different variations of an address to improve geocoding success."""
        if pd.isna(address) or not address.strip():
            return None
        
        variations = [
            address,  # Original
            address.replace('Local', '').replace('local', ''),  # Remove "Local"
            address.split(',')[0] if ',' in address else address,  # Just street part
            address.replace('#', '').replace('Apt', '').replace('Suite', ''),  # Remove unit info
        ]
        
        # Add variation with just street + city if Quebec addresses
        if 'Québec' in address or 'Quebec' in address:
            parts = address.split(',')
            if len(parts) >= 2:
                variations.append(f"{parts[0].strip()}, Montréal, Québec")
        
        for i, variation in enumerate(variations):
            if self.interrupted:
                return None
                
            variation = variation.strip()
            if not variation:
                continue
                
            logger.info(f"  Trying variation {i+1}: {variation}")
            coords = self.geocode_address(variation)
            if coords:
                logger.info(f"  ✓ Success with variation {i+1}!")
                return coords
            
            # Small delay between variations
            time.sleep(0.5)
        
        return None
    
    def fill_missing_gps(self, gps_file: str, address_column: str, max_attempts: int = 100):
        """Fill missing GPS coordinates using address variations and better geocoding."""
        logger.info(f"🔧 Attempting to fill missing GPS data in {gps_file}")
        logger.info(f"Will try up to {max_attempts} addresses")
        
        # Load the GPS file
        df = pd.read_csv(gps_file)
        
        # Find records missing GPS
        missing_mask = df['gps'].isna() | (df['gps'] == '')
        missing_df = df[missing_mask].copy()
        
        if len(missing_df) == 0:
            logger.info("No missing GPS data found!")
            return
        
        logger.info(f"Found {len(missing_df)} records missing GPS data")
        
        # Get unique addresses that are missing
        unique_missing_addresses = missing_df[address_column].dropna().unique()
        logger.info(f"Found {len(unique_missing_addresses)} unique addresses to geocode")
        
        # Limit attempts
        addresses_to_try = unique_missing_addresses[:max_attempts]
        logger.info(f"Will attempt {len(addresses_to_try)} addresses")
        
        address_to_coords = {}
        successful_fills = 0
        
        for i, address in enumerate(addresses_to_try):
            if self.interrupted:
                break
                
            logger.info(f"\n📍 Processing {i+1}/{len(addresses_to_try)}: {address}")
            
            # Try variations of the address
            coords = self.try_address_variations(address)
            
            if coords:
                address_to_coords[address] = f"{coords[0]},{coords[1]}"
                successful_fills += 1
                logger.info(f"✅ Success! GPS: {coords}")
            else:
                logger.warning("❌ All variations failed")
            
            # Save progress every 25 attempts
            if (i + 1) % 25 == 0:
                logger.info(f"💾 Checkpoint: Saving progress after {i+1} attempts...")
                self._apply_gps_updates(df, address_column, address_to_coords)
                df.to_csv(gps_file, index=False)
                logger.info(f"Current success rate: {successful_fills}/{i+1} = {(successful_fills/(i+1))*100:.1f}%")
        
        # Final save
        if not self.interrupted:
            logger.info("💾 Final save...")
            self._apply_gps_updates(df, address_column, address_to_coords)
            df.to_csv(gps_file, index=False)
            
            logger.info(f"✅ Completed! Successfully filled {successful_fills} addresses")
            logger.info(f"Final success rate: {successful_fills}/{len(addresses_to_try)} = {(successful_fills/len(addresses_to_try))*100:.1f}%")
        else:
            logger.info("❌ Interrupted by user")
    
    def _apply_gps_updates(self, df: pd.DataFrame, address_column: str, address_to_coords: Dict[str, str]):
        """Apply GPS coordinate updates to the dataframe."""
        for address, coords in address_to_coords.items():
            if coords:
                mask = df[address_column] == address
                df.loc[mask, 'gps'] = coords

def main():
    """Main function to run the gap analysis and filling process."""
    logger.info("🚀 Starting GPS Gap Filler and Analyzer")
    
    filler = GPSGapFiller(rate_limit_delay=1.5)
    
    try:
        # Analyze violations
        logger.info("\n" + "="*60)
        logger.info("ANALYZING VIOLATIONS DATA")
        logger.info("="*60)
        
        violations_analysis = filler.compare_and_analyze_gaps(
            'violations.csv', 
            'violations_with_gps.csv', 
            'adresse'
        )
        
        logger.info(f"📊 VIOLATIONS ANALYSIS RESULTS:")
        logger.info(f"  Total records: {violations_analysis['total_records']}")
        logger.info(f"  Successful geocodes: {violations_analysis['successful_geocodes']}")
        logger.info(f"  Failed geocodes: {violations_analysis['failed_geocodes']}")
        logger.info(f"  Success rate: {violations_analysis['success_rate']:.2f}%")
        logger.info(f"  Unique missing addresses: {violations_analysis['unique_missing_addresses']}")
        
        logger.info("🔍 Missing address patterns:")
        for pattern, count in violations_analysis['missing_patterns'].items():
            logger.info(f"  {pattern}: {count}")
        
        # Show sample missing addresses
        logger.info("📝 Sample missing addresses:")
        for i, addr in enumerate(violations_analysis['missing_addresses'][:5]):
            logger.info(f"  {i+1}. {addr}")
        
        # Analyze businesses
        logger.info("\n" + "="*60)
        logger.info("ANALYZING BUSINESSES DATA")
        logger.info("="*60)
        
        businesses_analysis = filler.compare_and_analyze_gaps(
            'businesses.csv', 
            'businesses_with_gps.csv', 
            'address'
        )
        
        logger.info(f"📊 BUSINESSES ANALYSIS RESULTS:")
        logger.info(f"  Total records: {businesses_analysis['total_records']}")
        logger.info(f"  Successful geocodes: {businesses_analysis['successful_geocodes']}")
        logger.info(f"  Failed geocodes: {businesses_analysis['failed_geocodes']}")
        logger.info(f"  Success rate: {businesses_analysis['success_rate']:.2f}%")
        logger.info(f"  Unique missing addresses: {businesses_analysis['unique_missing_addresses']}")
        
        logger.info("🔍 Missing address patterns:")
        for pattern, count in businesses_analysis['missing_patterns'].items():
            logger.info(f"  {pattern}: {count}")
        
        # Show sample missing addresses
        logger.info("📝 Sample missing addresses:")
        for i, addr in enumerate(businesses_analysis['missing_addresses'][:5]):
            logger.info(f"  {i+1}. {addr}")
        
        # Ask user which file to process first
        logger.info("\n" + "="*60)
        logger.info("FILLING MISSING GPS DATA")
        logger.info("="*60)
        
        logger.info("Ready to attempt filling missing GPS data.")
        logger.info("Press Ctrl+C at any time to stop gracefully.")
        
        # Fill violations first (smaller dataset)
        logger.info("\n🔧 Starting with violations (smaller dataset)...")
        filler.fill_missing_gps('violations_with_gps.csv', 'adresse', max_attempts=50)
        
        if not filler.interrupted:
            logger.info("\n🔧 Now processing businesses (this will take longer)...")
            filler.fill_missing_gps('businesses_with_gps.csv', 'address', max_attempts=100)
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
    finally:
        logger.info("🏁 GPS Gap Filler session ended")

if __name__ == "__main__":
    main()