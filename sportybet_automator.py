import time
import json
from typing import Dict, List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SportyBetAutomator:
    def __init__(self, headless: bool = True):
        """Initialize the automator with Chrome options"""
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 20)
        self.logged_in = False
        
    def login(self, phone_number: str, password: str) -> bool:
        """
        Login to SportyBet account
        Returns: True if successful, False otherwise
        """
        try:
            logger.info("Navigating to SportyBet...")
            self.driver.get("https://www.sportybet.com")
            
            # Wait for page to load and click login
            login_btn = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Sign In')]"))
            )
            login_btn.click()
            
            # Input phone number
            phone_input = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@type='tel' or contains(@placeholder, 'phone')]"))
            )
            phone_input.send_keys(phone_number)
            
            # Input password
            password_input = self.driver.find_element(By.XPATH, "//input[@type='password']")
            password_input.send_keys(password)
            
            # Submit login
            submit_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            submit_btn.click()
            
            # Wait for login to complete (check for account element)
            time.sleep(5)
            
            # Verify login by checking for account-related elements
            try:
                self.wait.until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'account')] | //span[contains(text(), 'My Account')]"))
                )
                self.logged_in = True
                logger.info("Login successful!")
                return True
            except:
                logger.error("Login verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            return False
    
    def get_available_games(self) -> List[Dict]:
        """
        Scrape available games/events
        Returns: List of game dictionaries
        """
        if not self.logged_in:
            logger.error("Not logged in!")
            return []
        
        games = []
        try:
            # Navigate to sports page
            self.driver.get("https://www.sportybet.com/sports")
            time.sleep(3)
            
            # Look for game elements (this will vary based on SportyBet's structure)
            game_elements = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'event') or contains(@class, 'match')]")
            
            for game in game_elements[:10]:  # Limit to 10 games
                try:
                    game_data = {
                        'title': game.text.split('\n')[0] if game.text else "Unknown",
                        'time': datetime.now().strftime("%H:%M"),
                        'odds': {},
                        'url': self.driver.current_url
                    }
                    games.append(game_data)
                except:
                    continue
            
            logger.info(f"Found {len(games)} games")
            return games
            
        except Exception as e:
            logger.error(f"Failed to get games: {str(e)}")
            return []
    
    def place_bet(self, game_url: str, selection: str, stake: float) -> bool:
        """
        Place a bet on a specific game
        Returns: True if successful, False otherwise
        """
        if not self.logged_in:
            logger.error("Not logged in!")
            return False
        
        try:
            # Navigate to specific game
            self.driver.get(game_url)
            time.sleep(3)
            
            # Find the selection (this is very site-specific)
            # You'll need to inspect SportyBet's HTML structure
            selection_btn = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '{selection}')]/ancestor::button"))
            )
            selection_btn.click()
            
            # Input stake
            stake_input = self.wait.until(
                EC.presence_of_element_located((By.XPATH, "//input[@type='number' and contains(@placeholder, 'stake')]"))
            )
            stake_input.clear()
            stake_input.send_keys(str(stake))
            
            # Place bet button
            place_bet_btn = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Place Bet')]"))
            )
            place_bet_btn.click()
            
            # Confirm bet if needed
            time.sleep(2)
            confirm_btn = self.driver.find_elements(By.XPATH, "//button[contains(text(), 'Confirm')]")
            if confirm_btn:
                confirm_btn[0].click()
            
            logger.info(f"Bet placed: {selection} with ${stake}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to place bet: {str(e)}")
            return False
    
    def send_to_telegram(self, games: List[Dict], telegram_bot_token: str, chat_id: str):
        """
        Send available games to Telegram channel
        """
        if not games:
            logger.warning("No games to send")
            return
        
        try:
            # Format message
            message = "🏆 *Available Games on SportyBet* 🏆\n\n"
            for i, game in enumerate(games, 1):
                message += f"{i}. {game['title']}\n"
                message += f"   ⏰ {game['time']}\n"
                message += f"   🔗 [View Game]({game['url']})\n\n"
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                logger.info("Games sent to Telegram successfully")
            else:
                logger.error(f"Failed to send to Telegram: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to send to Telegram: {str(e)}")
    
    def reasonable_betting_strategy(self, games: List[Dict], budget: float) -> Dict:
        """
        Basic reasonable betting strategy
        - Never bet more than 5% of budget
        - Focus on games with clear favorites
        - Avoid high-risk parlays
        """
        bet_suggestions = []
        
        for game in games:
            # Example simple strategy: bet on first 3 games with small stake
            if len(bet_suggestions) < 3:
                bet = {
                    'game': game['title'],
                    'selection': 'Home Win',  # This should be determined by analysis
                    'stake': min(budget * 0.02, 10),  # Max 2% or $10
                    'odds': 1.8  # Example odds
                }
                bet_suggestions.append(bet)
        
        return bet_suggestions
    
    def close(self):
        """Close the browser"""
        self.driver.quit()
        logger.info("Browser closed")


class BettingManager:
    def __init__(self, config_file: str = 'config.json'):
        """Load configuration from file"""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.automator = SportyBetAutomator(headless=True)
        
    def run(self):
        """Main execution loop"""
        try:
            # Login
            if not self.automator.login(
                self.config['sportybet']['phone'],
                self.config['sportybet']['password']
            ):
                logger.error("Failed to login. Exiting.")
                return
            
            # Get available games
            games = self.automator.get_available_games()
            
            # Send to Telegram
            self.automator.send_to_telegram(
                games,
                self.config['telegram']['bot_token'],
                self.config['telegram']['chat_id']
            )
            
            # Get betting suggestions
            suggestions = self.automator.reasonable_betting_strategy(
                games, 
                self.config['betting']['daily_budget']
            )
            
            logger.info(f"Betting suggestions: {suggestions}")
            
            # Optional: Place bets automatically
            if self.config['betting']['auto_place_bets']:
                for bet in suggestions:
                    # You'll need to implement proper game URL and selection mapping
                    # self.automator.place_bet(bet['game_url'], bet['selection'], bet['stake'])
                    pass
            
            # Wait and check periodically (example: every 30 minutes)
            # while True:
            #     time.sleep(1800)  # 30 minutes
            #     games = self.automator.get_available_games()
            #     # Process new games...
            
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
        finally:
            self.automator.close()


# Configuration file (config.json) should look like:
"""
{
    "sportybet": {
        "phone": "+1234567890",
        "password": "your_password"
    },
    "telegram": {
        "bot_token": "YOUR_BOT_TOKEN",
        "chat_id": "@your_channel_name"
    },
    "betting": {
        "daily_budget": 100.0,
        "auto_place_bets": false,
        "max_bet_percentage": 5,
        "min_odds": 1.5
    }
}
"""

if __name__ == "__main__":
    # Example usage
    manager = BettingManager('config.json')
    
    # For manual testing without Telegram
    # automator = SportyBetAutomator(headless=False)
    # automator.login("your_number", "your_password")
    # games = automator.get_available_games()
    # print(f"Found games: {len(games)}")
    # automator.close()
