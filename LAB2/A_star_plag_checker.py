import heapq
import re

# Preprocess text (normalize, tokenize into sentences)
def preprocess_text(text):
    # Normalize text: lowercasing and removing punctuation
    text = text.lower()
    
    # Replace all newlines with periods for consistent sentence separation
    text = text.replace('\n', '.')
    
    # Remove extra punctuation but leave sentence-ending periods
    text = re.sub(r'[^\w\s\.]', '', text)
    
    # Tokenize into sentences by splitting on periods
    sentences = text.split('.')
    
    # Remove any empty or purely whitespace sentences
    processed_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # Print separated sentences after preprocessing
    print("Preprocessed Sentences:\n")
    for i, sentence in enumerate(processed_sentences, 1):
        print(f"Sentence {i}: {sentence}")
    print("\n")  # Newline for separation
    
    return processed_sentences

# Levenshtein distance calculation (edit distance between two sentences)
def levenshtein_distance(s1, s2):
    dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    
    # Initialize the DP table with base cases
    for i in range(len(s1) + 1):
        for j in range(len(s2) + 1):
            if i == 0:
                dp[i][j] = j  # Cost of insertions
            elif j == 0:
                dp[i][j] = i  # Cost of deletions
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No cost if characters are the same
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])  # Min cost for insert, delete, substitute
                
    return dp[len(s1)][len(s2)]

# A* search node structure
class AStarNode:
    def __init__(self, pos1, pos2, g_cost, h_cost, parent=None):
        self.pos1 = pos1  # Index in Document 1
        self.pos2 = pos2  # Index in Document 2
        self.g_cost = g_cost  # Accumulated cost (g)
        self.h_cost = h_cost  # Heuristic cost (h)
        self.parent = parent  # Parent node for traceback
    
    def f_cost(self):
        return self.g_cost + self.h_cost
    
    # Comparator for the priority queue (min-heap)
    def __lt__(self, other):
        return self.f_cost() < other.f_cost()

def heuristic(pos1, pos2, sentences1, sentences2):
    # Use the average edit distance of remaining sentences as a heuristic
    remaining1 = len(sentences1) - pos1
    remaining2 = len(sentences2) - pos2
    if remaining1 == 0 or remaining2 == 0:
        return max(remaining1, remaining2) * 10  # Penalize for unaligned sentences
    return (remaining1 + remaining2) / 2  # Average of remaining sentences


# A* search algorithm for sentence alignment
def a_star_search(sentences1, sentences2):
    # Initialize the priority queue (frontier) with the initial state
    initial_node = AStarNode(0, 0, 0, heuristic(0, 0, sentences1, sentences2))
    frontier = []
    heapq.heappush(frontier, initial_node)
    explored = set()
    
    while frontier:
        current_node = heapq.heappop(frontier)
        pos1, pos2 = current_node.pos1, current_node.pos2

        # Debugging print to track explored nodes
        print(f"Exploring node: Document 1 Position: {pos1}, Document 2 Position: {pos2}, g_cost: {current_node.g_cost}, h_cost: {current_node.h_cost}")
        
        # Goal: both documents are fully aligned
        if pos1 == len(sentences1) and pos2 == len(sentences2):
            alignment = []
            node = current_node
            while node:
                alignment.append((node.pos1, node.pos2))
                node = node.parent
            return alignment[::-1]  # Return in reverse order
        
        explored.add((pos1, pos2))
        
        # Generate possible successor states (transitions)
        for move in [(1, 1), (1, 0), (0, 1)]:
            new_pos1 = pos1 + move[0]
            new_pos2 = pos2 + move[1]
            
            if new_pos1 <= len(sentences1) and new_pos2 <= len(sentences2) and (new_pos1, new_pos2) not in explored:
                g_cost = current_node.g_cost
                if move == (1, 1):  # Align the current sentences
                    g_cost += levenshtein_distance(sentences1[pos1], sentences2[pos2])
                else:  # Skipping a sentence (penalty)
                    g_cost += 1  # Add a skip penalty
                
                # Calculate heuristic
                h_cost = heuristic(new_pos1, new_pos2, sentences1, sentences2)
                new_node = AStarNode(new_pos1, new_pos2, g_cost, h_cost, current_node)
                
                heapq.heappush(frontier, new_node)
    
    return None  # If no alignment found

# Perform sentence alignment between two documents
def align_documents(doc1, doc2):
    sentences1 = preprocess_text(doc1)
    sentences2 = preprocess_text(doc2)
    
    alignment = a_star_search(sentences1, sentences2)
    
    aligned_sentences = []
    for (i, j) in alignment:
        if i < len(sentences1) and j < len(sentences2):
            distance = levenshtein_distance(sentences1[i], sentences2[j])
            aligned_sentences.append((sentences1[i], sentences2[j], distance))
        elif i < len(sentences1):
            aligned_sentences.append((sentences1[i], None, None))  # No match in Document 2
        elif j < len(sentences2):
            aligned_sentences.append((None, sentences2[j], None))  # No match in Document 1
    
    return aligned_sentences

# Detect plagiarism: sentences with a low edit distance or high similarity
def detect_plagiarism(aligned_sentences, threshold=5, similarity_threshold=0.7):
    plagiarized_pairs = []
    
    for s1, s2, distance in aligned_sentences:
        if s1 and s2 and distance is not None:
            # Calculate percentage similarity based on edit distance and sentence lengths
            max_len = max(len(s1), len(s2))
            similarity = 1 - (distance / max_len)  # Similarity score between 0 and 1
            
            # Check for low edit distance (near match) or high percentage similarity
            if distance <= threshold or similarity >= similarity_threshold:
                plagiarized_pairs.append((s1, s2, distance, similarity))
    
    return plagiarized_pairs

# Load text from a file
def load_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Main function to load files and run the alignment and plagiarism detection
def main(doc1_path, doc2_path):
    doc1 = load_text_from_file(doc1_path)
    doc2 = load_text_from_file(doc2_path)

    aligned = align_documents(doc1, doc2)
    print("Aligned Sentences with Distances:\n")
    for s1, s2, distance in aligned:
        if s1 is None:
            print(f"Document 1: [No Sentence]\nDocument 2: {s2}\n")
        elif s2 is None:
            print(f"Document 1: {s1}\nDocument 2: [No Sentence]\n")
        else:
            print(f"Document 1: {s1}\nDocument 2: {s2}\nEdit Distance: {distance}\n")

    # Detecting potential plagiarism using both edit distance and similarity score
    plagiarism = detect_plagiarism(aligned, threshold=5, similarity_threshold=0.7)
    
    print("\nPotential Plagiarism (with Distance and Similarity):\n")
    for s1, s2, distance, similarity in plagiarism:
        print(f"Sentence 1: {s1}\nSentence 2: {s2}\nEdit Distance: {distance}\nSimilarity: {similarity:.2f}\n")

# Example usage (you can replace these file paths with actual text files)
doc1_path = "LAB2\\doc1.txt"
doc2_path = "LAB2\\doc2.txt"

main(doc1_path, doc2_path)
