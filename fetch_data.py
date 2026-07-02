import os

DATA_DIR = "data"

def ensure_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def create_mock_data():
    files = {
        "Character_Bios.txt": """Harry Potter: The Boy Who Lived. Born July 31, 1980. Son of James and Lily Potter.
Hermione Granger: Brightest witch of her age. Born September 19, 1979. Muggle-born.
Ron Weasley: Harry's best friend. Born March 1, 1980. Pure-blood.
Albus Dumbledore: Headmaster of Hogwarts. Defeated Grindelwald in 1945.
Tom Riddle (Voldemort): The Dark Lord. Heir of Slytherin. Created Horcruxes to achieve immortality.""",
        
        "Timeline.txt": """1981: Voldemort attacks the Potters in Godric's Hollow.
1991: Harry discovers he is a wizard and attends Hogwarts.
1992: The Chamber of Secrets is opened.
1993: Sirius Black escapes Azkaban.
1994: The Triwizard Tournament takes place at Hogwarts. Voldemort returns.
1995: The Order of the Phoenix is reformed.
1996: Battle of the Department of Mysteries.
1997: Dumbledore dies. The Ministry falls.
1998: Battle of Hogwarts. Voldemort is defeated.""",
        
        "Lore.txt": """Horcruxes: Objects in which a Dark wizard has hidden a fragment of his soul in order to become immortal. Voldemort created seven: Tom Riddle's Diary, Marvolo Gaunt's Ring, Salazar Slytherin's Locket, Helga Hufflepuff's Cup, Rowena Ravenclaw's Diadem, Harry Potter (accidental), and Nagini the snake.
Deathly Hallows: Three highly powerful magical objects created by Death: the Elder Wand, the Resurrection Stone, and the Cloak of Invisibility.""",
        
        "Deleted_Scenes_and_Notes.txt": """1. Philosopher's Stone:
Deleted Scene: Peeves the Poltergeist. Rik Mayall was cast and filmed scenes as Peeves, but they were cut.
Production Note: The floating candles in the Great Hall were initially real candles.

2. Chamber of Secrets:
Deleted Scene: Lucius Malfoy and Draco in Borgin and Burkes.
Production Note: The basilisk was a full-sized animatronic.

3. Goblet of Fire:
Deleted Scene: A longer version of the Yule Ball featuring the Weird Sisters.

4. Deathly Hallows Part 2:
Deleted Scene: Remus Lupin and Nymphadora Tonks sharing a quiet moment before the Battle of Hogwarts."""
    }
    
    for filename, content in files.items():
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created: {filepath}")

if __name__ == "__main__":
    ensure_dir()
    create_mock_data()
    print("Mock data creation complete.")
