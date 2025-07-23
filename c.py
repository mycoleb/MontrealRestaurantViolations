import pandas as pd
import requests
import time
import json
from typing import Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CanadaGPSValidator:
    def __init__(self):
        """
        Initialize the GPS validator with Canada's approximate boundaries
        """
        # Canada's approximate bounding box
        self.canada_bounds = {
            'north': 83.1,    # Northernmost point
            'south': 41.7,    # Southernmost point (Point Pelee, Ontario)
            'east': -52.6,    # Easternmost point (Newfoundland)
            'west': -141.0    # Westernmost point (Yukon border with Alaska)
        }
        
        # Rate limiting for API calls
        self.api_delay = 1.0  # seconds between API calls
        
    def is_in_canada(self, lat: float, lon: float) -> bool:
        """
        Check if coordinates are within Canada's approximate boundaries
        """
        return (self.canada_bounds['south'] <= lat <= self.canada_bounds['north'] and
                self.canada_bounds['west'] <= lon <= self.canada_bounds['east'])
    
    def parse_gps_string(self, gps_string: str) -> Optional[Tuple[float, float]]:
        """
        Parse GPS string to latitude and longitude
        """
        if not gps_string or str(gps_string).lower() in ['nan', 'none', '']:
            return None
            
        try:
            gps_str = str(gps_string).strip()
            if ',' in gps_str:
                parts = gps_str.split(',')
                if len(parts) >= 2:
                    lat = float(parts[0].strip())
                    lon = float(parts[1].strip())
                    return lat, lon
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def get_coordinates_from_address(self, address: str, city: str = "", province: str = "") -> Optional[Tuple[float, float]]:
        """
        Get coordinates from address using multiple geocoding services
        """
        # Try different geocoding services
        coords = self._try_nominatim(address, city, province)
        if coords:
            return coords
            
        # Could add more services here (Google Maps API, etc.)
        # coords = self._try_google_maps(address, city, province)
        
        return None
    
    def _try_nominatim(self, address: str, city: str = "", province: str = "") -> Optional[Tuple[float, float]]:
        """
        Try geocoding with OpenStreetMap Nominatim (free service)
        """
        try:
            # Construct search query
            query_parts = [address]
            if city:
                query_parts.append(city)
            if province:
                query_parts.append(province)
            query_parts.append("Canada")
            
            full_address = ", ".join(filter(None, query_parts))
            
            # Nominatim API endpoint
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': full_address,
                'format': 'json',
                'countrycodes': 'ca',  # Restrict to Canada
                'limit': 1,
                'addressdetails': 1
            }
            
            headers = {
                'User-Agent': 'GPS-Validator-Script/1.0'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                
                # Verify the result is actually in Canada
                if self.is_in_canada(lat, lon):
                    logger.info(f"Found coordinates for '{full_address}': {lat}, {lon}")
                    return lat, lon
                else:
                    logger.warning(f"Geocoded coordinates for '{full_address}' are outside Canada: {lat}, {lon}")
            
            # Rate limiting
            time.sleep(self.api_delay)
            
        except Exception as e:
            logger.error(f"Error geocoding address '{address}': {str(e)}")
        
        return None
    
    def _try_google_maps(self, address: str, city: str = "", province: str = "") -> Optional[Tuple[float, float]]:
        """
        Try geocoding with Google Maps API (requires API key)
        Uncomment and configure if you have a Google Maps API key
        """
        # GOOGLE_MAPS_API_KEY = "your_api_key_here"
        # 
        # try:
        #     query_parts = [address]
        #     if city:
        #         query_parts.append(city)
        #     if province:
        #         query_parts.append(province)
        #     query_parts.append("Canada")
        #     
        #     full_address = ", ".join(filter(None, query_parts))
        #     
        #     url = "https://maps.googleapis.com/maps/api/geocode/json"
        #     params = {
        #         'address': full_address,
        #         'key': GOOGLE_MAPS_API_KEY,
        #         'region': 'ca'
        #     }
        #     
        #     response = requests.get(url, params=params, timeout=10)
        #     response.raise_for_status()
        #     
        #     data = response.json()
        #     if data['status'] == 'OK' and data['results']:
        #         location = data['results'][0]['geometry']['location']
        #         lat = location['lat']
        #         lon = location['lng']
        #         
        #         if self.is_in_canada(lat, lon):
        #             return lat, lon
        #     
        #     time.sleep(self.api_delay)
        #     
        # except Exception as e:
        #     logger.error(f"Error with Google Maps geocoding: {str(e)}")
        
        return None
    
    def process_businesses_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Process the businesses CSV file
        """
        logger.info(f"Processing businesses file: {input_file}")
        
        df = pd.read_csv(input_file)
        stats = {
            'total_records': len(df),
            'invalid_coords': 0,
            'outside_canada': 0,
            'corrected': 0,
            'failed_corrections': 0
        }
        
        # Add new columns for corrected coordinates
        df['original_gps'] = df['gps'].copy()
        df['corrected'] = False
        df['correction_method'] = ''
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processed {idx}/{len(df)} business records")
            
            # Parse GPS coordinates
            coords = self.parse_gps_string(row['gps'])
            
            if coords is None:
                stats['invalid_coords'] += 1
                # Try to get coordinates from address
                new_coords = self.get_coordinates_from_address(
                    str(row['address']), str(row['city']), str(row['state'])
                )
                if new_coords:
                    lat, lon = new_coords
                    df.at[idx, 'gps'] = f"{lat},{lon}"
                    df.at[idx, 'latitude'] = lat
                    df.at[idx, 'longitude'] = lon
                    df.at[idx, 'corrected'] = True
                    df.at[idx, 'correction_method'] = 'invalid_coords_geocoded'
                    stats['corrected'] += 1
                else:
                    stats['failed_corrections'] += 1
                continue
            
            lat, lon = coords
            
            # Check if coordinates are in Canada
            if not self.is_in_canada(lat, lon):
                stats['outside_canada'] += 1
                logger.info(f"Coordinates outside Canada for business {row['name']}: {lat}, {lon}")
                
                # Try to get correct coordinates from address
                new_coords = self.get_coordinates_from_address(
                    str(row['address']), str(row['city']), str(row['state'])
                )
                
                if new_coords:
                    new_lat, new_lon = new_coords
                    df.at[idx, 'gps'] = f"{new_lat},{new_lon}"
                    df.at[idx, 'latitude'] = new_lat
                    df.at[idx, 'longitude'] = new_lon
                    df.at[idx, 'corrected'] = True
                    df.at[idx, 'correction_method'] = 'outside_canada_geocoded'
                    stats['corrected'] += 1
                    logger.info(f"Corrected coordinates: {new_lat}, {new_lon}")
                else:
                    stats['failed_corrections'] += 1
                    logger.warning(f"Failed to correct coordinates for: {row['address']}, {row['city']}")
        
        # Save the corrected file
        df.to_csv(output_file, index=False)
        logger.info(f"Saved corrected businesses file: {output_file}")
        
        return stats
    
    def process_violations_file(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """
        Process the violations CSV file
        """
        logger.info(f"Processing violations file: {input_file}")
        
        df = pd.read_csv(input_file)
        stats = {
            'total_records': len(df),
            'invalid_coords': 0,
            'outside_canada': 0,
            'corrected': 0,
            'failed_corrections': 0
        }
        
        # Add new columns for corrected coordinates
        df['original_gps'] = df['gps'].copy()
        df['corrected'] = False
        df['correction_method'] = ''
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"Processed {idx}/{len(df)} violation records")
            
            # Parse GPS coordinates
            coords = self.parse_gps_string(row['gps'])
            
            if coords is None:
                stats['invalid_coords'] += 1
                # Try to get coordinates from address
                new_coords = self.get_coordinates_from_address(
                    str(row['adresse']), str(row['ville']), ""
                )
                if new_coords:
                    lat, lon = new_coords
                    df.at[idx, 'gps'] = f"{lat},{lon}"
                    df.at[idx, 'corrected'] = True
                    df.at[idx, 'correction_method'] = 'invalid_coords_geocoded'
                    stats['corrected'] += 1
                else:
                    stats['failed_corrections'] += 1
                continue
            
            lat, lon = coords
            
            # Check if coordinates are in Canada
            if not self.is_in_canada(lat, lon):
                stats['outside_canada'] += 1
                logger.info(f"Coordinates outside Canada for violation at {row['adresse']}: {lat}, {lon}")
                
                # Try to get correct coordinates from address
                new_coords = self.get_coordinates_from_address(
                    str(row['adresse']), str(row['ville']), ""
                )
                
                if new_coords:
                    new_lat, new_lon = new_coords
                    df.at[idx, 'gps'] = f"{new_lat},{new_lon}"
                    df.at[idx, 'corrected'] = True
                    df.at[idx, 'correction_method'] = 'outside_canada_geocoded'
                    stats['corrected'] += 1
                    logger.info(f"Corrected coordinates: {new_lat}, {new_lon}")
                else:
                    stats['failed_corrections'] += 1
                    logger.warning(f"Failed to correct coordinates for: {row['adresse']}, {row['ville']}")
        
        # Save the corrected file
        df.to_csv(output_file, index=False)
        logger.info(f"Saved corrected violations file: {output_file}")
        
        return stats
    
    def generate_report(self, business_stats: Dict[str, Any], violation_stats: Dict[str, Any]):
        """
        Generate a summary report
        """
        print("\n" + "="*60)
        print("GPS COORDINATES VALIDATION AND CORRECTION REPORT")
        print("="*60)
        
        print("\nBUSINESSES:")
        print(f"  Total records: {business_stats['total_records']:,}")
        print(f"  Invalid coordinates: {business_stats['invalid_coords']:,}")
        print(f"  Outside Canada: {business_stats['outside_canada']:,}")
        print(f"  Successfully corrected: {business_stats['corrected']:,}")
        print(f"  Failed corrections: {business_stats['failed_corrections']:,}")
        
        print("\nVIOLATIONS:")
        print(f"  Total records: {violation_stats['total_records']:,}")
        print(f"  Invalid coordinates: {violation_stats['invalid_coords']:,}")
        print(f"  Outside Canada: {violation_stats['outside_canada']:,}")
        print(f"  Successfully corrected: {violation_stats['corrected']:,}")
        print(f"  Failed corrections: {violation_stats['failed_corrections']:,}")
        
        total_issues = (business_stats['invalid_coords'] + business_stats['outside_canada'] + 
                       violation_stats['invalid_coords'] + violation_stats['outside_canada'])
        total_corrected = business_stats['corrected'] + violation_stats['corrected']
        
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total coordinate issues found: {total_issues:,}")
        print(f"  Total successfully corrected: {total_corrected:,}")
        if total_issues > 0:
            success_rate = (total_corrected / total_issues) * 100
            print(f"  Success rate: {success_rate:.1f}%")
        
        print("\nOUTPUT FILES:")
        print("  - businesses_with_gps_corrected.csv")
        print("  - violations_with_gps_corrected.csv")
        print("\nNew columns added:")
        print("  - original_gps: Original GPS coordinates")
        print("  - corrected: Boolean indicating if coordinates were corrected")
        print("  - correction_method: Method used for correction")

def main():
    """
    Main function to run the GPS validation and correction process
    """
    print("GPS Coordinates Validator and Corrector for Canada")
    print("=" * 50)
    
    validator = CanadaGPSValidator()
    
    try:
        # Process businesses file
        business_stats = validator.process_businesses_file(
            'businesses_with_gps.csv', 
            'businesses_with_gps_corrected.csv'
        )
        
        # Process violations file
        violation_stats = validator.process_violations_file(
            'violations_with_gps.csv', 
            'violations_with_gps_corrected.csv'
        )
        
        # Generate report
        validator.generate_report(business_stats, violation_stats)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find input file - {e}")
        print("Make sure 'businesses_with_gps.csv' and 'violations_with_gps.csv' are in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    # Install required packages
    print("Make sure you have the required packages installed:")
    print("pip install pandas requests")
    print()
    
    main()