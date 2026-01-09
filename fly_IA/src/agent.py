import ollama
import json
from scrapper import WebScraper

class FlightAgent:
    def __init__(self, model="llama3.1"):
        """
        Initialise l'agent conversationnel pour les vols
        
        Args:
            model (str): Nom du modèle Ollama à utiliser
        """
        self.model = model
        self.scraper = WebScraper()
        self.conversation_history = []
        
        # Mapping des aéroports français
        self.airports_mapping = {
            "paris": ["parisaeroport.fr", "aeroportparisbeauvais.com"],
            "nice": ["nice.aeroport.fr"],
            "marseille": ["marseille.aeroport.fr"],
            "reunion": ["reunion.aeroport.fr"],
            "rennes": ["rennes.aeroport.fr"]
        }
        
        # Définition des outils disponibles pour l'agent
        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'search_flights',
                    'description': 'Recherche des informations de vols entre deux villes. Retourne les horaires, numéros de vol et statuts.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'departure': {
                                'type': 'string',
                                'description': 'Ville ou aéroport de départ (ex: Paris, Rennes, Nice)',
                            },
                            'destination': {
                                'type': 'string',
                                'description': 'Ville ou aéroport de destination (ex: Paris, Marseille)',
                            },
                        },
                        'required': ['departure', 'destination'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_airport_info',
                    'description': 'Récupère les informations générales d\'un aéroport',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'airport': {
                                'type': 'string',
                                'description': 'Nom de la ville ou de l\'aéroport',
                            },
                        },
                        'required': ['airport'],
                    },
                },
            },
        ]
    
    def search_flights(self, departure, destination):
        """
        Recherche des vols entre deux villes
        
        Args:
            departure (str): Ville de départ
            destination (str): Ville de destination
            
        Returns:
            dict: Résultats de la recherche
        """
        print(f"🔍 Recherche de vols: {departure} → {destination}")
        
        # Normalisation des noms de villes
        departure = departure.lower()
        destination = destination.lower()
        
        # Construction des URLs à scraper
        urls_to_scrape = []
        
        # Ajouter les URLs des aéroports de départ
        if departure in self.airports_mapping:
            for domain in self.airports_mapping[departure]:
                urls_to_scrape.append(f"https://{domain}/vols-departs")
                urls_to_scrape.append(f"https://{domain}/")
        
        # Utiliser Skyscanner pour recherche globale
        skyscanner_url = f"https://www.skyscanner.fr/transport/vols/{departure}/{destination}"
        urls_to_scrape.append(skyscanner_url)
        
        # Scraper les URLs
        results = []
        for url in urls_to_scrape:
            scrape_result = self.scraper.scrape_url(url)
            if scrape_result['success']:
                results.append({
                    'url': url,
                    'content': scrape_result['content'][:1000],  # Limiter la taille
                    'title': scrape_result['title']
                })
        
        return {
            'success': True,
            'departure': departure,
            'destination': destination,
            'results': results,
            'message': f"Trouvé {len(results)} sources d'informations"
        }
    
    def get_airport_info(self, airport):
        """
        Récupère les informations d'un aéroport
        
        Args:
            airport (str): Nom de l'aéroport ou de la ville
            
        Returns:
            dict: Informations de l'aéroport
        """
        print(f"ℹ️  Récupération d'infos pour: {airport}")
        
        airport = airport.lower()
        
        if airport in self.airports_mapping:
            url = f"https://{self.airports_mapping[airport][0]}/"
            result = self.scraper.scrape_url(url)
            
            if result['success']:
                return {
                    'success': True,
                    'airport': airport,
                    'info': result['content'][:800]
                }
        
        return {
            'success': False,
            'message': f"Aéroport '{airport}' non trouvé dans la liste"
        }
    
    def execute_function(self, function_name, arguments):
        """
        Exécute une fonction appelée par le modèle
        
        Args:
            function_name (str): Nom de la fonction
            arguments (dict): Arguments de la fonction
            
        Returns:
            dict: Résultat de l'exécution
        """
        if function_name == "search_flights":
            return self.search_flights(
                arguments.get('departure'),
                arguments.get('destination')
            )
        elif function_name == "get_airport_info":
            return self.get_airport_info(arguments.get('airport'))
        else:
            return {'success': False, 'error': 'Fonction inconnue'}
    
    def chat(self, user_message):
        """
        Gère une conversation avec l'utilisateur
        
        Args:
            user_message (str): Message de l'utilisateur
            
        Returns:
            str: Réponse de l'agent
        """
        print(f"\n👤 Utilisateur: {user_message}")
        
        # Ajouter le message à l'historique
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        
        # Préparer le prompt système
        system_prompt = """Tu es un assistant spécialisé dans la recherche de vols aériens en France.
Tu peux rechercher des informations sur les vols entre différentes villes françaises.
Tu dois être précis et utile. Quand tu utilises les outils, analyse bien les résultats pour donner une réponse claire.
Réponds toujours en français."""
        
        # Appel à Ollama avec les outils
        response = ollama.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                *self.conversation_history
            ],
            tools=self.tools,
        )
        
        # Traiter la réponse
        if response['message'].get('tool_calls'):
            # Le modèle veut utiliser un outil
            for tool_call in response['message']['tool_calls']:
                function_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']
                
                print(f"🔧 Appel de fonction: {function_name}")
                print(f"   Arguments: {arguments}")
                
                # Exécuter la fonction
                function_result = self.execute_function(function_name, arguments)
                
                # Ajouter le résultat à l'historique
                self.conversation_history.append({
                    'role': 'tool',
                    'content': json.dumps(function_result, ensure_ascii=False)
                })
            
            # Re-appeler le modèle avec les résultats des outils
            final_response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    *self.conversation_history
                ]
            )
            
            assistant_message = final_response['message']['content']
        else:
            # Réponse directe sans outil
            assistant_message = response['message']['content']
        
        # Ajouter la réponse à l'historique
        self.conversation_history.append({
            'role': 'assistant',
            'content': assistant_message
        })
        
        print(f"🤖 Assistant: {assistant_message}")
        
        return assistant_message


# Test de l'agent
if __name__ == "__main__":
    print("=== Test de l'Agent de Recherche de Vols ===\n")
    
    agent = FlightAgent()
    
    # Test 1
    agent.chat("Quels sont les vols de Paris vers Nice ?")
    
    # Test 2
    print("\n" + "="*50 + "\n")
    agent.chat("Donne-moi des informations sur l'aéroport de Marseille")