import json
from urllib.parse import urlparse
from pathlib import Path

class URLValidator:
    def __init__(self, config_path="config/allowed_sites.json"):
        """Initialise le validateur avec la liste des domaines autorisés"""
        self.config_path = Path(config_path)
        self.allowed_domains = self._load_allowed_domains()
    
    def _load_allowed_domains(self):
        """Charge la liste des domaines autorisés depuis le fichier JSON"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('allowed_domains', [])
        except FileNotFoundError:
            print(f"⚠️  Fichier de configuration non trouvé: {self.config_path}")
            return []
        except json.JSONDecodeError:
            print(f"⚠️  Erreur de lecture du fichier JSON")
            return []
    
    def is_url_allowed(self, url):
        """
        Vérifie si une URL est autorisée selon la whitelist
        
        Args:
            url (str): L'URL à vérifier
            
        Returns:
            bool: True si l'URL est autorisée, False sinon
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Enlève le 'www.' si présent
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Vérifie si le domaine ou un sous-domaine est autorisé
            for allowed_domain in self.allowed_domains:
                allowed_domain = allowed_domain.lower()
                if allowed_domain.startswith('www.'):
                    allowed_domain = allowed_domain[4:]
                
                # Vérifie correspondance exacte ou sous-domaine
                if domain == allowed_domain or domain.endswith('.' + allowed_domain):
                    return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  Erreur lors de la validation de l'URL: {e}")
            return False
    
    def filter_urls(self, urls):
        """
        Filtre une liste d'URLs pour ne garder que celles autorisées
        
        Args:
            urls (list): Liste d'URLs à filtrer
            
        Returns:
            list: Liste des URLs autorisées
        """
        return [url for url in urls if self.is_url_allowed(url)]
    
    def get_allowed_domains(self):
        """Retourne la liste des domaines autorisés"""
        return self.allowed_domains


# Test du validateur
if __name__ == "__main__":
    validator = URLValidator()
    
    # Tests d'exemple
    test_urls = [
        "https://www.skyscanner.fr",
        "https://www.google.com",
        "https://parisaeroport.fr",
    ]
    
    print("🔍 Test du validateur d'URLs\n")
    print(f"Domaines autorisés: {validator.get_allowed_domains()}\n")
    
    for url in test_urls:
        is_allowed = validator.is_url_allowed(url)
        status = "✅ AUTORISÉ" if is_allowed else "❌ BLOQUÉ"
        print(f"{status}: {url}")