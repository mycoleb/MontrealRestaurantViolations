import pandas as pd
import requests
import time
import signal
import sys
from typing import Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InterruptibleGeocoder:
    def __init__(self, rate_limit_delay: float = 2.0):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AddressGeocoder/1.0'})
        self.interrupted = False
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.info("\n🛑 Interrupt received! Stopping gracefully...")
        logger.info("Saving progress and cleaning up...")
        self.interrupted = True
        
        # Give it a moment to clean up
        time.sleep(1)
        sys.exit(0)
    
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode with interrupt checking."""
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
            
            # Use shorter timeout to make it more responsive
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 503:
                logger.warning("Service unavailable - API overloaded")
                return None
            elif response.status_code == 429:
                logger.warning("Rate limited - waiting longer")
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
                time.sleep(0.1)  # Sleep in small chunks
    
    def process_csv_with_checkpoints(self, input_file: str, output_file: str, address_column: str, checkpoint_interval: int = 100):
        """Process CSV with regular checkpoints for recovery."""
        logger.info(f"Processing {input_file} with checkpoints every {checkpoint_interval} addresses...")
        
        try:
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} rows")
            
            if address_column not in df.columns:
                logger.error(f"Column '{address_column}' not found")
                return
            
            # Initialize GPS column if it doesn't exist
            if 'gps' not in df.columns:
                df['gps'] = None
            
            # Get unique addresses that haven't been processed
            unique_addresses = df[address_column].dropna().unique()
            processed_count = 0
            
            logger.info(f"Found {len(unique_addresses)} unique addresses")
            logger.info("Press Ctrl+C at any time to stop gracefully")
            
            address_to_coords = {}
            
            for i, address in enumerate(unique_addresses):
                if self.interrupted:
                    break
                    
                logger.info(f"Processing {i+1}/{len(unique_addresses)}: {address}")
                
                coords = self.geocode_address(address)
                if coords:
                    address_to_coords[address] = f"{coords[0]},{coords[1]}"
                    logger.info(f"✓ Success: {coords}")
                else:
                    address_to_coords[address] = None
                    logger.warning("✗ Failed")
                
                processed_count += 1
                
                # Save checkpoint
                if processed_count % checkpoint_interval == 0:
                    logger.info(f"💾 Checkpoint: Saving progress after {processed_count} addresses...")
                    
                    # Apply current mapping
                    df['gps'] = df[address_column].map(address_to_coords)
                    df.to_csv(output_file, index=False)
                    
                    success_rate = len([v for v in address_to_coords.values() if v is not None]) / len(address_to_coords) * 100
                    logger.info(f"Current success rate: {success_rate:.1f}%")
            
            # Final save
            if not self.interrupted:
                logger.info("💾 Final save...")
                df['gps'] = df[address_column].map(address_to_coords)
                df.to_csv(output_file, index=False)
                logger.info(f"✅ Completed! Saved to {output_file}")
            else:
                logger.info("❌ Interrupted by user")
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")

def main():
    """Main function with graceful interrupt handling."""
    logger.info("🚀 Starting interruptible geocoder...")
    logger.info("Press Ctrl+C at any time to stop gracefully")
    
    geocoder = InterruptibleGeocoder(rate_limit_delay=1.0)  # 1-second delay
    
    try:
        # Process violations.csv with checkpoints every 50 addresses
        geocoder.process_csv_with_checkpoints(
            input_file='violations.csv',
            output_file='violations_with_gps.csv',
            address_column='adresse',
            checkpoint_interval=50
        )
        
        if not geocoder.interrupted:
            # Process businesses.csv
            geocoder.process_csv_with_checkpoints(
                input_file='businesses.csv',
                output_file='businesses_with_gps.csv',
                address_column='address',
                checkpoint_interval=50
            )
            
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
    finally:
        logger.info("🏁 Geocoding session ended")

if __name__ == "__main__":
    main()