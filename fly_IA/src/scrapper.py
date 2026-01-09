import requests
from bs4 import BeautifulSoup
import time

try:
    from .url_validator import URLValidator
except ImportError:
    from url_validator import URLValidator

class WebScraper:
    def __init__(self, config_path="config/allowed_sites.json"):
        """Initialise le scraper avec validation d'URLs"""
        self.validator = URLValidator(config_path)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.delay = 1  # Délai entre les requêtes en secondes
    
    def scrape_url(self, url):
        """
        Scrape le contenu d'une URL si elle est autorisée
        
        Args:
            url (str): L'URL à scraper
            
        Returns:
            dict: Contenu scrapé avec statut et données
        """
        # Vérification de l'URL
        if not self.validator.is_url_allowed(url):
            return {
                'success': False,
                'error': 'URL non autorisée',
                'url': url
            }
        
        try:
            # Respect du rate limiting
            time.sleep(self.delay)
            
            # Requête HTTP
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse le HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extraction du contenu
            # Retire les scripts et styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Récupère le texte
            text = soup.get_text(separator=' ', strip=True)
            
            # Récupère le titre
            title = soup.title.string if soup.title else "Sans titre"
            
            return {
                'success': True,
                'url': url,
                'title': title,
                'content': text,
                'status_code': response.status_code
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Erreur de requête: {str(e)}',
                'url': url
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Erreur de scraping: {str(e)}',
                'url': url
            }
    
    def scrape_multiple_urls(self, urls):
        """
        Scrape plusieurs URLs
        
        Args:
            urls (list): Liste d'URLs à scraper
            
        Returns:
            list: Liste des résultats de scraping
        """
        results = []
        for url in urls:
            print(f"🔍 Scraping: {url}")
            result = self.scrape_url(url)
            results.append(result)
            
            if result['success']:
                print(f"✅ Succès: {result['title']}")
            else:
                print(f"❌ Échec: {result['error']}")
        
        return results
    
    def extract_flight_info(self, html_content, url=""):
        """
        Extrait les informations de vol depuis le contenu HTML
        Adapté pour les sites d'aéroports français
        
        Args:
            html_content (str): Contenu HTML de la page
            url (str): URL source pour identifier le site
            
        Returns:
            list: Liste des informations de vols trouvées
        """
        soup = BeautifulSoup(html_content, 'lxml')
        flights = []
        
        # Recherche multi-critères pour différents formats
        
        # 1. Tables de vols (format classique)
        tables = soup.find_all('table', class_=lambda x: x and any(
            keyword in str(x).lower() for keyword in 
            ['flight', 'vol', 'depart', 'arrival', 'arrivee']
        ))
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Skip header
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    flight_data = {
                        'flight_number': cells[0].get_text(strip=True),
                        'destination': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                        'time': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                        'status': cells[3].get_text(strip=True) if len(cells) > 3 else 'N/A'
                    }
                    flights.append(flight_data)
        
        # 2. Divs ou sections de vols (format moderne)
        flight_divs = soup.find_all(['div', 'section', 'article'], class_=lambda x: x and any(
            keyword in str(x).lower() for keyword in 
            ['flight-card', 'vol-item', 'flight-row', 'departure', 'arrival']
        ))
        
        for div in flight_divs:
            flight_data = {}
            
            # Recherche des éléments dans le div
            number_elem = div.find(class_=lambda x: x and 'number' in str(x).lower())
            dest_elem = div.find(class_=lambda x: x and any(
                k in str(x).lower() for k in ['destination', 'city', 'ville']
            ))
            time_elem = div.find(class_=lambda x: x and any(
                k in str(x).lower() for k in ['time', 'heure', 'hour']
            ))
            status_elem = div.find(class_=lambda x: x and 'status' in str(x).lower())
            
            if dest_elem or time_elem:
                flight_data = {
                    'flight_number': number_elem.get_text(strip=True) if number_elem else 'N/A',
                    'destination': dest_elem.get_text(strip=True) if dest_elem else 'N/A',
                    'time': time_elem.get_text(strip=True) if time_elem else 'N/A',
                    'status': status_elem.get_text(strip=True) if status_elem else 'N/A'
                }
                flights.append(flight_data)
        
        return flights
    
    def search_flights(self, departure=None, destination=None, date=None):
        """
        Recherche des vols selon des critères
        
        Args:
            departure (str): Ville/aéroport de départ
            destination (str): Ville/aéroport d'arrivée
            date (str): Date du vol
            
        Returns:
            list: Vols trouvés correspondant aux critères
        """
        # Cette fonction sera utilisée par l'agent pour chercher des vols spécifiques
        results = []
        
        # Construire les URLs à scraper selon les critères
        urls_to_check = []
        
        # Logique pour construire les URLs appropriées
        # À adapter selon vos besoins
        
        return results


# Test du scraper
if __name__ == "__main__":
    scraper = WebScraper()
    
    # Test avec une URL
    test_url = "https://www.rennes.aeroport.fr"
    result = scraper.scrape_url(test_url)
    
    if result['success']:
        print(f"\n✅ Scraping réussi!")
        print(f"Titre: {result['title']}")
        print(f"Contenu (100 premiers caractères): {result['content'][:100]}...")
    else:
        print(f"\n❌ Échec: {result['error']}")