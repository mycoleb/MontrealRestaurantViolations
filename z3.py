import pandas as pd
import requests
import time
import logging
from typing import Optional, Tuple, Dict, List
import re
from difflib import get_close_matches

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AddressFixerGeocode:
    def __init__(self, rate_limit_delay: float = 1.0):
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'AddressFixer/1.0'})
        
        # Define address abbreviation expansions
        self.abbreviation_map = {
            # French abbreviations commonly used in Quebec
            'Av.': 'Avenue',
            'Boul.': 'Boulevard', 
            'Ch.': 'Chemin',
            'Pl.': 'Place',
            'Sq.': 'Square',
            'St.': 'Saint',
            'Ste.': 'Sainte',
            'R.': 'Rue',
            'Blvd.': 'Boulevard',
            'Dr.': 'Drive',
            'Cres.': 'Crescent',
            'Pkwy.': 'Parkway',
            'Terr.': 'Terrace',
            'Ct.': 'Court',
            'Cir.': 'Circle',
            'Mt.': 'Mount'
        }
        
        # Define Montreal area cities for fallback geocoding
        self.montreal_area_cities = [
            'Montréal', 'Montreal',
            'Laval', 'Longueuil', 'Brossard', 'Saint-Laurent', 'Dollard-des-Ormeaux',
            'Kirkland', 'Pointe-Claire', 'Dorval', 'Lachine', 'LaSalle', 'Verdun',
            'Westmount', 'Mont-Royal', 'Outremont', 'Côte Saint-Luc', 'Hampstead',
            'Montreal-Nord', 'Montréal-Nord', 'Saint-Léonard', 'Anjou', 'Pierrefonds',
            'Roxboro', 'Sainte-Anne-de-Bellevue', 'Beaconsfield', "Baie-D'Urfé",
            'Senneville', "L'Île-Bizard", 'Sainte-Geneviève'
        ]
        
        # Former municipalities now part of Montreal - replace with "Montreal" if geocoding fails
        self.former_montreal_municipalities = {
            'St-Léonard': 'Montreal',
            'Saint-Léonard': 'Montreal',
            'St-Laurent': 'Montreal', 
            'Saint-Laurent': 'Montreal',
            'Anjou': 'Montreal',
            'Pierrefonds': 'Montreal',
            'Roxboro': 'Montreal',
            'Montreal-Nord': 'Montreal',
            'Montréal-Nord': 'Montreal',
            'LaSalle': 'Montreal',
            'Lachine': 'Montreal',
            'Verdun': 'Montreal',
            'Outremont': 'Montreal',
            'Westmount': 'Montreal',
            'Mont-Royal': 'Montreal',
            'Côte Saint-Luc': 'Montreal',
            'Hampstead': 'Montreal',
            'Dollard-des-Ormeaux': 'Montreal',
            'Dollard-Des-Ormeaux': 'Montreal',
            'Pointe-Claire': 'Montreal',
            'Kirkland': 'Montreal',
            'Beaconsfield': 'Montreal',
            'Sainte-Anne-de-Bellevue': 'Montreal',
            "Baie-D'Urfé": 'Montreal',
            'Senneville': 'Montreal',
            "L'Île-Bizard": 'Montreal',
            'Sainte-Geneviève': 'Montreal'
        }
    
    def expand_abbreviations(self, address: str) -> str:
        """Expand common address abbreviations."""
        if not address:
            return address
            
        expanded = address
        for abbrev, full in self.abbreviation_map.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            expanded = re.sub(pattern, full, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def clean_address(self, address: str) -> str:
        """Clean and standardize address format."""
        if not address:
            return address
            
        cleaned = address.strip()
        
        # Remove extra spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # FIXED: Handle Quebec-specific address formatting issues correctly
        # Remove letters attached to street numbers (e.g., "7064A Boul." -> "7064 Boul.", "5198A Boulevard" -> "5198 Boulevard")
        cleaned = re.sub(r'^(\d+)[A-Z](\s+)', r'\1\2', cleaned, flags=re.IGNORECASE)
        
        # Remove nonsense words that appear between numbers and street names
        # Pattern: number + nonsense_word + street_type
        street_types = r'\b(Rue|Boulevard|Boul\.|Avenue|Av\.|Chemin|Ch\.|Place|Pl\.|Street|St\.?|Road|Rd\.?)\b'
        cleaned = re.sub(r'^(\d+)\s+[a-z]{3,8}\s+(' + street_types + r')', r'\1 \2', cleaned, flags=re.IGNORECASE)
        
        # FIXED: Handle Quebec hyphenated street names better
        # "Charles-De La Tour" should become "Charles-de-LaTour" (remove spaces around hyphens in street names)
        # But only for parts that look like street names (after Rue, Boulevard, etc.)
        def fix_hyphenated_names(match):
            street_type = match.group(1)
            street_name = match.group(2)
            # Remove spaces around hyphens and title case inconsistencies
            fixed_name = re.sub(r'\s*-\s*', '-', street_name)
            # Fix common Quebec street name patterns
            fixed_name = re.sub(r'-De\s+La\s+', '-de-La', fixed_name, flags=re.IGNORECASE)
            fixed_name = re.sub(r'-Du\s+', '-du-', fixed_name, flags=re.IGNORECASE)
            fixed_name = re.sub(r'-Des\s+', '-des-', fixed_name, flags=re.IGNORECASE)
            return f"{street_type} {fixed_name}"
        
        cleaned = re.sub(r'(Rue|Boulevard|Boul\.|Avenue|Av\.|Chemin|Ch\.|Place|Pl\.)\s+([^,]+)', 
                        fix_hyphenated_names, cleaned, flags=re.IGNORECASE)
        
        # Remove unit numbers with letters in the middle of addresses 
        # (e.g., "5100 42-B Boulevard" -> "5100 Boulevard", "1432 6B Boulevard" -> "1432 Boulevard")
        # But be more careful to avoid removing legitimate street numbers
        cleaned = re.sub(r'\s+\d+[A-Z]\s+(?=(?:Rue|Boulevard|Boul\.|Avenue|Av\.|Chemin|Ch\.|Place|Pl\.)\s)', ' ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+\d+-[A-Z]\s+(?=(?:Rue|Boulevard|Boul\.|Avenue|Av\.|Chemin|Ch\.|Place|Pl\.)\s)', ' ', cleaned, flags=re.IGNORECASE)
        
        # Remove complex unit identifiers like "TR-CK-05"
        cleaned = re.sub(r'\s+[A-Z]{2,}-[A-Z]{2,}-\d+\s+', ' ', cleaned, flags=re.IGNORECASE)
        
        # Remove "Local" and unit information that often confuses geocoders
        # Handle Quebec-specific patterns first: "Local RC-07", "Local 123A", "Local A-1", etc.
        cleaned = re.sub(r',?\s*Local\s+[A-Z0-9-]+[A-Z0-9]?(?=,|$)', '', cleaned, flags=re.IGNORECASE)
        # Then handle general local patterns
        cleaned = re.sub(r',?\s*Local\s+[^,]*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r',?\s*Suite\s+[^,]*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r',?\s*Unit\s+[^,]*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r',?\s*Apt\s+[^,]*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r',?\s*#[^,]*', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up any resulting double commas or spaces
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip(' ,')
        
        return cleaned
    
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address with error handling."""
        if pd.isna(address) or not address.strip():
            return None
            
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': address,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1,
                'countrycodes': 'ca'  # Limit to Canada for better results
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 503:
                logger.warning("Service unavailable - waiting...")
                time.sleep(5)
                return None
            elif response.status_code == 429:
                logger.warning("Rate limited - waiting longer...")
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
            time.sleep(self.rate_limit_delay)
    
    def has_city_in_address(self, address: str) -> bool:
        """Check if address already contains a city name."""
        if not address:
            return False
        
        address_lower = address.lower()
        # Check for explicit city indicators
        city_indicators = ['montréal', 'montreal', 'laval', 'longueuil', 'brossard', 'dorval', 
                          'lachine', 'lasalle', 'verdun', 'westmount', 'québec', 'quebec',
                          'st-léonard', 'saint-léonard', 'st-laurent', 'saint-laurent', 'anjou',
                          'pierrefonds', 'roxboro', 'montreal-nord', 'montréal-nord', 'outremont',
                          'mont-royal', 'côte saint-luc', 'hampstead', 'dollard-des-ormeaux',
                          'pointe-claire', 'kirkland', 'beaconsfield', 'sainte-anne-de-bellevue',
                          "baie-d'urfé", 'senneville', "l'île-bizard", 'sainte-geneviève']
        
        return any(city in address_lower for city in city_indicators)
    
    def try_former_municipality_replacement(self, address: str) -> Optional[Tuple[float, float]]:
        """Try replacing former Montreal municipalities with 'Montreal' in the address."""
        if not address:
            return None

        logger.info(f"  🏛️ Trying former municipality replacements for: {address}")

        found_replacements = []
        modified_address = address

        for former_city, replacement in self.former_montreal_municipalities.items():
            pattern = r'\b' + re.escape(former_city) + r'\b'
            if re.search(pattern, address, re.IGNORECASE):
                found_replacements.append((former_city, replacement))
                modified_address = re.sub(pattern, replacement, modified_address, flags=re.IGNORECASE)

        if found_replacements:
            for former, replacement in found_replacements:
                logger.info(f"    Replacing '{former}' with '{replacement}'")
            
            logger.info(f"    Modified address: {modified_address}")
            coords = self.geocode_address(modified_address)
            if coords:
                logger.info(f"    ✅ SUCCESS with municipality replacement!")
                return coords
            else:
                logger.info(f"    ❌ Failed with municipality replacement")

        return None


    def try_cities_fallback(self, base_address: str) -> Optional[Tuple[float, float]]:
        """Try the same address with different Montreal area cities."""
        if not base_address:
            return None

        cleaned_base = self.expand_abbreviations(self.clean_address(base_address))
        
        # Remove any city/province/country info
        street_part = re.sub(
            r',\s*(Montréal|Montreal|St-Laurent|Saint-Laurent|Laval|Longueuil|Québec|Quebec|QC|Canada).*',
            '',
            cleaned_base,
            flags=re.IGNORECASE
        )

        logger.info(f"  🏙️ Trying different cities for: {base_address}")

        priority_cities = ['Dorval', 'Montréal', 'Laval', 'Saint-Laurent', 'Longueuil', 'Brossard']
        other_cities = [city for city in self.montreal_area_cities if city not in priority_cities]
        all_cities_to_try = priority_cities + other_cities

        for i, city in enumerate(all_cities_to_try[:8]):
            city_address = f"{street_part}, {city}, Quebec, Canada"
            logger.info(f"    City {i+1}: {city_address}")

            coords = self.geocode_address(city_address)
            if coords:
                logger.info(f"    ✅ SUCCESS with {city}!")
                return coords
            else:
                logger.info(f"    ❌ Failed with {city}")

        return None

    def fuzzy_match_known_addresses(self, failed_address: str, df: pd.DataFrame, address_column: str) -> Optional[Tuple[float, float]]:
        """Try to find GPS coordinates by fuzzy matching to addresses that already have coordinates."""
        if not failed_address:
            return None
        
        # Get all addresses that already have GPS coordinates
        successful_addresses = df[df['gps'].notna() & (df['gps'] != '')][address_column].dropna().unique()
        
        if len(successful_addresses) == 0:
            return None
        
        # Clean the failed address for better matching
        clean_failed = self.clean_address(failed_address).lower()
        clean_successful = [self.clean_address(addr).lower() for addr in successful_addresses]
        
        # Find close matches
        close_matches = get_close_matches(clean_failed, clean_successful, n=3, cutoff=0.8)
        
        if close_matches:
            logger.info(f"  🔍 Found fuzzy matches for '{failed_address}':")
            for i, match in enumerate(close_matches):
                # Find the original address that corresponds to this clean match
                for orig_addr in successful_addresses:
                    if self.clean_address(orig_addr).lower() == match:
                        # Get the GPS coordinates for this address
                        gps_row = df[(df[address_column] == orig_addr) & (df['gps'].notna())].iloc[0]
                        gps_coords = gps_row['gps']
                        if gps_coords and ',' in str(gps_coords):
                            try:
                                lat, lon = map(float, str(gps_coords).split(','))
                                logger.info(f"    Match {i+1}: '{orig_addr}' -> GPS: {lat:.6f},{lon:.6f}")
                                logger.info(f"    ✅ Using fuzzy match!")
                                return (lat, lon)
                            except ValueError:
                                continue
                        break
        
        return None
    
    def try_address_variations(self, original_address: str, postal_codes: Dict[str, str] = None, df: pd.DataFrame = None, address_column: str = None) -> Optional[Tuple[float, float, int]]:
        """Try multiple variations of an address to improve geocoding success."""
        if not original_address:
            return None
        
        # Create variations with enhanced Quebec-specific cleaning
        variations = [
            original_address,  # Original first
            self.expand_abbreviations(original_address),  # Expand abbreviations
            self.clean_address(original_address),  # Clean version
            self.expand_abbreviations(self.clean_address(original_address)),  # Both
        ]
        
        # Add extra variations for Quebec-specific issues
        cleaned_base = self.clean_address(original_address)
        expanded_base = self.expand_abbreviations(cleaned_base)
        
        # Get postal code for this address if available
        postal_code = ""
        if postal_codes and original_address in postal_codes:
            postal_code = postal_codes[original_address]
            logger.info(f"  Using postal code: {postal_code}")
        
        # Add variations with city context for Quebec addresses
        if any(word in original_address.lower() for word in ['québec', 'quebec', 'montréal', 'montreal']):
            variations.extend([
                f"{expanded_base}, Montréal, Québec, Canada",
                f"{expanded_base}, Montreal, Quebec, Canada",
                f"{expanded_base}, QC, Canada",
                f"{cleaned_base}, Montréal, QC",
                f"{expanded_base}, Montréal"
            ])
            if postal_code:
                variations.extend([
                    f"{expanded_base}, Montréal, QC, {postal_code}",
                    f"{expanded_base}, Montreal, Quebec, {postal_code}",
                    f"{cleaned_base}, {postal_code}"
                ])
        else:
            # For addresses without explicit city, assume Montreal first
            variations.extend([
                f"{expanded_base}, Montréal, Québec, Canada",
                f"{expanded_base}, Montreal, Quebec, Canada",
                f"{cleaned_base}, Montréal, QC"
            ])
            if postal_code:
                variations.extend([
                    f"{expanded_base}, Montréal, QC, {postal_code}",
                    f"{expanded_base}, Montreal, Quebec, {postal_code}",
                    f"{cleaned_base}, {postal_code}"
                ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var and var not in seen:
                unique_variations.append(var)
                seen.add(var)
        
        logger.info(f"Trying {len(unique_variations)} variations for: {original_address}")
        
        for i, variation in enumerate(unique_variations):
            logger.info(f"  Variation {i+1}: {variation}")
            coords = self.geocode_address(variation)
            if coords:
                logger.info(f"  ✅ SUCCESS with variation {i+1}!")
                return (coords[0], coords[1], i+1)  # Return coordinates and which variation worked
            else:
                logger.info(f"  ❌ Failed")
        
        # After the first 4 main variations fail, try former municipality replacement
        if not self.has_city_in_address(original_address):
            logger.info(f"  🏙️ No city detected, trying Montreal area cities...")
            city_result = self.try_cities_fallback(original_address)
            if city_result:
                return (city_result[0], city_result[1], 888)  # Use 888 to indicate city fallback
        else:
            # If address has a city, try former municipality replacement first
            logger.info(f"  🏛️ Trying former municipality replacements...")
            municipality_result = self.try_former_municipality_replacement(original_address)
            if municipality_result:
                return (municipality_result[0], municipality_result[1], 777)  # Use 777 to indicate municipality replacement
            
            # Then try city fallback as backup
            logger.info(f"  🏙️ Trying Montreal area cities as backup...")
            city_result = self.try_cities_fallback(original_address)
            if city_result:
                return (city_result[0], city_result[1], 888)  # Use 888 to indicate city fallback
        
        # If all variations failed, try fuzzy matching as a last resort
        if df is not None and address_column is not None:
            logger.info(f"  🔍 Trying fuzzy matching as last resort...")
            fuzzy_result = self.fuzzy_match_known_addresses(original_address, df, address_column)
            if fuzzy_result:
                return (fuzzy_result[0], fuzzy_result[1], 999)  # Use 999 to indicate fuzzy match
        
        return None
    
    def extract_postal_code_from_data(self, df: pd.DataFrame, address_column: str) -> Dict[str, str]:
        """Extract Canadian postal codes from the dataset if available."""
        postal_codes = {}
        
        # Check if there are columns that might contain postal codes
        potential_columns = [col for col in df.columns if any(word in col.lower() for word in ['postal', 'zip', 'code', 'cp'])]
        
        if potential_columns:
            logger.info(f"Found potential postal code columns: {potential_columns}")
            for col in potential_columns:
                for _, row in df.iterrows():
                    address = row[address_column]
                    postal = row[col]
                    if pd.notna(address) and pd.notna(postal):
                        # Canadian postal code pattern: A1A 1A1 or A1A1A1
                        if re.match(r'^[A-Z]\d[A-Z]\s?\d[A-Z]\d$', str(postal).upper().strip()):
                            postal_codes[address] = str(postal).upper().strip()
        
        # Also check if postal codes are embedded in addresses
        for _, row in df.iterrows():
            address = row[address_column]
            if pd.notna(address):
                # Look for Canadian postal codes in the address string
                postal_match = re.search(r'\b([A-Z]\d[A-Z]\s?\d[A-Z]\d)\b', str(address).upper())
                if postal_match:
                    postal_code = postal_match.group(1)
                    # Standardize format with space
                    if len(postal_code) == 6:
                        postal_code = postal_code[:3] + ' ' + postal_code[3:]
                    postal_codes[address] = postal_code
        
        logger.info(f"Found {len(postal_codes)} addresses with Canadian postal codes")
        return postal_codes
    
    def analyze_missing_addresses(self, df: pd.DataFrame, address_column: str) -> Dict:
        """Analyze patterns in addresses missing GPS coordinates."""
        missing_df = df[df['gps'].isna() | (df['gps'] == '')]
        
        if len(missing_df) == 0:
            return {'total_missing': 0}
        
        addresses = missing_df[address_column].dropna().tolist()
        
        patterns = {
            'total_missing': len(missing_df),
            'unique_addresses': len(set(addresses)),
            'contains_av': sum(1 for addr in addresses if 'Av.' in str(addr)),
            'contains_boul': sum(1 for addr in addresses if 'Boul.' in str(addr)),
            'contains_local': sum(1 for addr in addresses if 'Local' in str(addr)),
            'contains_ch': sum(1 for addr in addresses if 'Ch.' in str(addr)),
            'has_unit_info': sum(1 for addr in addresses if any(x in str(addr).lower() for x in ['local', 'suite', 'apt', '#'])),
            'has_nonsense_words': sum(1 for addr in addresses if re.search(r'\d+\s+[a-z]{3,8}\s+(rue|boulevard|avenue|boul\.|av\.)', str(addr), re.IGNORECASE)),
            'sample_addresses': addresses[:10]
        }
        
        return patterns
    
    def fill_missing_gps(self, input_file: str, output_file: str, address_column: str, max_attempts: int = 200):
        """Fill missing GPS coordinates using improved geocoding."""
        logger.info(f"🔧 Filling missing GPS data in {input_file}")
        
        # Load the file
        df = pd.read_csv(input_file)
        original_count = len(df)
        
        # Extract postal codes from the dataset
        postal_codes = self.extract_postal_code_from_data(df, address_column)
        
        # FIRST: Count current missing GPS to get accurate total
        missing_mask = df['gps'].isna() | (df['gps'] == '')
        current_missing_count = missing_mask.sum()
        logger.info(f"📊 Current missing GPS count: {current_missing_count}")
        
        # Get unique addresses that need geocoding
        unique_missing_addresses = df[missing_mask][address_column].dropna().unique()
        total_unique_missing = len(unique_missing_addresses)
        
        logger.info(f"📊 Unique addresses needing geocoding: {total_unique_missing}")
        
        # Analyze current gaps
        analysis = self.analyze_missing_addresses(df, address_column)
        logger.info(f"📊 Missing GPS Analysis:")
        for key, value in analysis.items():
            if key != 'sample_addresses':
                logger.info(f"  {key}: {value}")
        
        logger.info("📝 Sample missing addresses:")
        for i, addr in enumerate(analysis.get('sample_addresses', [])[:5]):
            logger.info(f"  {i+1}. {addr}")
        
        logger.info(f"Will attempt {min(max_attempts, total_unique_missing)} addresses")
        
        # Limit attempts
        addresses_to_process = unique_missing_addresses[:max_attempts]
        
        successful_geocodes = {}
        success_count = 0
        variation_success_stats = {}  # Track which variation number succeeds
        
        for i, address in enumerate(addresses_to_process):
            logger.info(f"\n📍 Attempting {i+1}/{len(addresses_to_process)}: Processing address")
            
            result = self.try_address_variations(address, postal_codes, df, address_column)
            
            if result:
                lat, lon, variation_num = result
                successful_geocodes[address] = f"{lat},{lon}"
                success_count += 1
                
                # Track which variation succeeded
                if variation_num not in variation_success_stats:
                    variation_success_stats[variation_num] = 0
                variation_success_stats[variation_num] += 1
                
                if variation_num == 999:
                    logger.info(f"✅ SUCCESS! GPS: {lat:.6f},{lon:.6f} (Fuzzy Match)")
                elif variation_num == 888:
                    logger.info(f"✅ SUCCESS! GPS: {lat:.6f},{lon:.6f} (City Fallback)")
                elif variation_num == 777:
                    logger.info(f"✅ SUCCESS! GPS: {lat:.6f},{lon:.6f} (Municipality Replacement)")
                else:
                    logger.info(f"✅ SUCCESS! GPS: {lat:.6f},{lon:.6f} (Variation {variation_num})")
            else:
                logger.warning("❌ All variations failed")
            
            # Save progress every 25 addresses
            if (i + 1) % 25 == 0:
                logger.info(f"💾 Checkpoint: Saving progress...")
                self._apply_updates(df, address_column, successful_geocodes)
                df.to_csv(output_file, index=False)
                current_rate = (success_count / (i + 1)) * 100
                logger.info(f"Current success rate: {success_count}/{i+1} = {current_rate:.1f}%")
        
        # Final application of all successful geocodes
        logger.info("💾 Applying final updates...")
        self._apply_updates(df, address_column, successful_geocodes)
        
        # Save final result
        df.to_csv(output_file, index=False)
        
        # Final statistics
        final_missing = len(df[df['gps'].isna() | (df['gps'] == '')])
        filled_count = current_missing_count - final_missing
        
        logger.info(f"✅ COMPLETED!")
        logger.info(f"  Started with missing: {current_missing_count}")
        logger.info(f"  Successfully filled: {filled_count}")
        logger.info(f"  Still missing: {final_missing}")
        logger.info(f"  Attempted addresses: {len(addresses_to_process)}")
        logger.info(f"  Success rate: {(success_count/len(addresses_to_process))*100:.1f}%")
        logger.info(f"  Saved to: {output_file}")
        
        # Report variation success statistics
        if variation_success_stats:
            logger.info(f"\n📈 VARIATION SUCCESS STATISTICS:")
            total_successes = sum(variation_success_stats.values())
            for variation_num in sorted(variation_success_stats.keys()):
                count = variation_success_stats[variation_num]
                percentage = (count / total_successes) * 100
                if variation_num == 999:
                    logger.info(f"  Fuzzy Match: {count} successes ({percentage:.1f}%)")
                elif variation_num == 888:
                    logger.info(f"  City Fallback: {count} successes ({percentage:.1f}%)")
                elif variation_num == 777:
                    logger.info(f"  Municipality Replacement: {count} successes ({percentage:.1f}%)")
                else:
                    logger.info(f"  Variation {variation_num}: {count} successes ({percentage:.1f}%)")
            
            # Identify which variations never work
            regular_variations = [v for v in variation_success_stats.keys() if v not in [777, 888, 999]]
            max_variation = max(regular_variations) if regular_variations else 0
            unused_variations = [v for v in range(1, max_variation + 1) if v not in variation_success_stats]
            if unused_variations:
                logger.info(f"  Variations that never succeeded: {unused_variations}")
                
            # Report on municipality replacement effectiveness
            if 777 in variation_success_stats:
                municipality_count = variation_success_stats[777]
                logger.info(f"  🏛️ Municipality replacement rescued {municipality_count} addresses with former Montreal municipalities!")
                
            # Report on city fallback effectiveness
            if 888 in variation_success_stats:
                city_count = variation_success_stats[888]
                logger.info(f"  🏙️ City fallback rescued {city_count} addresses missing city names!")
                
            # Report on fuzzy matching effectiveness
            if 999 in variation_success_stats:
                fuzzy_count = variation_success_stats[999]
                logger.info(f"  🎯 Fuzzy matching rescued {fuzzy_count} addresses that failed all other methods!")
        else:
            logger.info(f"\n📈 No successful variations to analyze")
        
        return df
    
    def _apply_updates(self, df: pd.DataFrame, address_column: str, geocode_results: Dict[str, str]):
        """Apply geocoding results to the dataframe."""
        for address, gps_coords in geocode_results.items():
            mask = df[address_column] == address
            df.loc[mask, 'gps'] = gps_coords

def main():
    """Main function to process both files."""
    logger.info("🚀 Starting Address Abbreviation Fixer and GPS Gap Filler")
    logger.info("This tool specifically handles French abbreviations like 'Av.', 'Boul.', etc.")
    
    fixer = AddressFixerGeocode(rate_limit_delay=1.0)
    
    try:
        # Process violations first (smaller dataset)
        logger.info("\n" + "="*70)
        logger.info("PROCESSING VIOLATIONS")
        logger.info("="*70)
        
        violations_df = fixer.fill_missing_gps(
            input_file='violations_with_gps.csv',
            output_file='violations_with_gps.csv',
            address_column='adresse',
            max_attempts=100
        )
        
        # Process businesses
        logger.info("\n" + "="*70)
        logger.info("PROCESSING BUSINESSES")
        logger.info("="*70)
        
        businesses_df = fixer.fill_missing_gps(
            input_file='businesses_with_gps.csv',
            output_file='businesses_with_gps.csv',
            address_column='address',
            max_attempts=150
        )
        
        logger.info("\n🎉 ALL PROCESSING COMPLETE!")
        logger.info("Both files have been updated with improved GPS coordinates.")
        
    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

    
    
    
    
    
    
    
    
    
def main():
    """Main function to process both files."""
    logger.info("🚀 Starting Address Abbreviation Fixer and GPS Gap Filler")
    logger.info("This tool specifically handles French abbreviations like 'Av.', 'Boul.', etc.")
    
    fixer = AddressFixerGeocode(rate_limit_delay=1.0)
    
    try:
        # Process violations first (smaller dataset)
        logger.info("\n" + "="*70)
        logger.info("PROCESSING VIOLATIONS")
        logger.info("="*70)
        
        violations_df = fixer.fill_missing_gps(
            input_file='violations_with_gps.csv',
            output_file='violations_with_gps.csv',
            address_column='adresse',
            max_attempts=100
        )
        
        # Process businesses
        logger.info("\n" + "="*70)
        logger.info("PROCESSING BUSINESSES")
        logger.info("="*70)
        
        businesses_df = fixer.fill_missing_gps(
            input_file='businesses_with_gps.csv',
            output_file='businesses_with_gps.csv',
            address_column='address',
            max_attempts=150
        )
        
        logger.info("\n🎉 ALL PROCESSING COMPLETE!")
        logger.info("Both files have been updated with improved GPS coordinates.")
        
    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()