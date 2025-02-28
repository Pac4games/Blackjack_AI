import cv2
import warnings
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

def blackjack_strategy(player_hand, dealer_upcard):
    # Calculate the player's total
    total = calculate_total(player_hand)
    is_soft = is_soft_hand(player_hand)
    is_pair = is_pair_func(player_hand)

    # Determine the dealer's upcard value
    dealer_value = card_value(dealer_upcard)

    # Apply the strategy
    if is_pair:
        return handle_pairs(player_hand, dealer_value)
    elif is_soft:
        return handle_soft_totals(total, dealer_value)
    else:
        return handle_hard_totals(total, dealer_value)

def calculate_total(hand):
    total = 0
    aces = 0
    for card in hand:
        value = card_value(card)
        if value == 11:  # Ace
            aces += 1
        total += value
    # Adjust for aces if total > 21
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total

def is_soft_hand(hand):
    return 11 in [card_value(card) for card in hand] and calculate_total(hand) <= 21

def is_pair_func(hand):
    return len(hand) == 2 and card_value(hand[0]) == card_value(hand[1])

def card_value(card):
    # Extract the numeric part of the card (e.g., '8S' -> '8', 'AH' -> 'A')
    value = card[:-1]  # Remove the last character (suit)

    # Convert face cards and Ace to their corresponding values
    if value in ['J', 'Q', 'K']:
        return 10
    elif value == 'A':
        return 11
    else:
        return int(value)  # Convert numeric cards to integers

def handle_pairs(hand, dealer_value):
    pair_card = card_value(hand[0])
    if pair_card == 11:  # Aces
        return "Split"
    elif pair_card == 10:  # 10s
        return "Stand"
    elif pair_card == 9 and dealer_value not in [7, 10, 11]:  # 9s
        return "Split"
    elif pair_card == 8:  # 8s
        return "Split"
    elif pair_card == 7 and dealer_value <= 7:  # 7s
        return "Split"
    elif pair_card == 6 and dealer_value <= 6:  # 6s
        return "Split"
    elif pair_card == 5:  # 5s
        return "Double Down" if dealer_value <= 9 else "Hit"
    elif pair_card == 4:  # 4s
        return "Hit"
    else:
        return "Hit"

def handle_soft_totals(total, dealer_value):
    if total >= 19:
        return "Stand"
    elif total == 18:
        return "Stand" if dealer_value <= 8 else "Hit"
    else:
        return "Hit"

def handle_hard_totals(total, dealer_value):
    if total >= 17:
        return "Stand"
    elif total >= 13 and dealer_value <= 6:
        return "Stand"
    elif total == 12 and dealer_value <= 3:
        return "Hit"
    elif total == 11:
        return "Double Down" if dealer_value <= 10 else "Hit"
    elif total == 10:
        return "Double Down" if dealer_value <= 9 else "Hit"
    elif total == 9:
        return "Double Down" if dealer_value <= 6 else "Hit"
    else:
        return "Hit"


# Load the trained model
model_path = '../best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Open the camera feed
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform inference on the grayscale frame
    results = model(gray_frame)

    # Parse the results
    detections = results.pandas().xyxy[0]  # Detections in pandas DataFrame format
    player_hand = []
    dealer_upcard = None

    for _, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        label = detection['name']
        confidence = detection['confidence']

        # Draw bounding box and label on the colored frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Extract player hand and dealer upcard
        x_center = (x1 + x2) / 2
        if x_center < frame.shape[1] / 2:  # Player cards
            player_hand.append(label)
        else:  # Dealer card
            dealer_upcard = label

    # Get the recommended action
    if player_hand and dealer_upcard:
        action = blackjack_strategy(player_hand, dealer_upcard)
        cv2.putText(frame, f"Action: {action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the colored frame with detections
    cv2.imshow("Blackjack Assistant", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("Blackjack Assistant", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
