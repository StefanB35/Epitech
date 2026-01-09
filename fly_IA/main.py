import sys
import os

# Ajouter le dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Maintenant les imports fonctionneront
import agent
from agent import FlightAgent

def print_header():
    """Affiche l'en-tête de l'application"""
    print("=" * 60)
    print("🛫 AGENT DE RECHERCHE DE VOLS - AÉROPORTS FRANÇAIS 🛬")
    print("=" * 60)
    print("\nVilles disponibles:")
    print("  • Paris (Charles de Gaulle, Orly, Beauvais)")
    print("  • Nice")
    print("  • Marseille")
    print("  • La Réunion")
    print("  • Rennes")
    print("\nExemples de questions:")
    print("  - Quel est le prochain vol de Paris vers Nice ?")
    print("  - Donne-moi les vols au départ de Marseille")
    print("  - Informations sur l'aéroport de La Réunion")
    print("\nCommandes:")
    print("  • 'quit' ou 'exit' pour quitter")
    print("  • 'clear' pour effacer l'historique")
    print("=" * 60)
    print()

def main():
    """Fonction principale de l'application"""
    print_header()
    
    # Initialiser l'agent
    try:
        print("🔄 Initialisation de l'agent avec Ollama (llama3.1:8b)...")
        flight_agent = FlightAgent(model="llama3.1:8b")
        print("✅ Agent initialisé avec succès!\n")
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        print("\n💡 Assurez-vous qu'Ollama est bien lancé.")
        return
    
    # Boucle principale de conversation
    while True:
        try:
            # Demander l'input utilisateur
            user_input = input("Vous: ").strip()
            
            # Vérifier les commandes spéciales
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Au revoir! Bon voyage!")
                break
            
            if user_input.lower() == 'clear':
                flight_agent.conversation_history = []
                print("\n🗑️  Historique effacé!\n")
                continue
            
            if not user_input:
                continue
            
            # Traiter la requête
            print()
            response = flight_agent.chat(user_input)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir! Bon voyage!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}\n")
            continue

if __name__ == "__main__":
    main()