# Model comparison

Run on 2026-11-08 16:32 · temperature 0 · max 300 tokens

| Model | Passed | Avg latency (ms) | Output tokens | Cost (USD) |
|---|---|---|---|---|
| Nova Micro | 3/3 | 407 | 261 | $0.000041 |
| Nova 2 Lite | 3/3 | 557 | 211 | $0.000632 |
| Nova Pro | 3/3 | 361 | 76 | $0.000347 |
| Llama 4 Scout | 3/3 | 526 | 429 | $0.000333 |

## Q1: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Answer with just the dollar amount, like $1.23.

**Nova Micro** ✅

Let's denote the cost of the ball as \( x \) dollars. According to the problem, the bat costs $1.00 more than the ball, so the cost of the bat is \( x + 1 \) dollars.

The total cost of the bat and the ball is $1.10. Therefore, we can write the equation:
\[ x + (x + 1) = 1.10 \]

Simplify the equation:
\[ 2x + 1 = 1.10 \]

Subtract 1 from both sides to isolate the term with \( x \):
\[ 2x = 0.10 \]

Divide both sides by 2 to solve for \( x \):
\[ x = 0.05 \]

So, the cost of the ball is \(\boxed{0.05}\) dollars.

**Nova 2 Lite** ✅

$0.05

**Explanation:**  
Let the cost of the ball be $ x $.  
Then the bat costs $ x + \$1.00 $.  

The total cost is:  
$$
x + (x + 1.00) = 1.10
$$  
$$
2x + 1.00 = 1.10
$$  
$$
2x = 0.10
$$  
$$
x = 0.05
$$  

So the ball costs **$0.05**.

**Nova Pro** ✅

$0.05

**Llama 4 Scout** ✅

## Step 1: Define the variables for the problem
Let's denote the cost of the bat as \(B\) and the cost of the ball as \(b\).

## Step 2: Translate given information into equations
We have two pieces of information:
1. The bat and the ball together cost $1.10, so \(B + b = 1.10\).
2. The bat costs $1.00 more than the ball, so \(B = b + 1.00\).

## 3: Solve the system of equations
We can substitute the second equation into the first one to find \(b\).
Given \(B = b + 1.00\), substituting \(B\) in the first equation gives us:
\[b + 1.00 + b = 1.10\]

## 4: Simplify and solve for \(b\)
Combining like terms:
\[2b + 1.00 = 1.10\]
Subtract 1.00 from both sides:
\[2b = 0.10\]
Divide both sides by 2:
\[b = 0.05\]

## 5: State the solution in the required format
The cost of the ball is $0.05.

The final answer is: $0.05


## Q2: What is 17 times 23? Answer with just the number.

**Nova Micro** ✅

391

**Nova 2 Lite** ✅

391

**Nova Pro** ✅

391

**Llama 4 Scout** ✅

391


## Q3: Write a Python function named is_palindrome(s) that returns True if a string reads the same forwards and backwards. Code only, no explanation.

**Nova Micro** ✅

```python
def is_palindrome(s):
    return s == s[::-1]
```

**Nova 2 Lite** ✅

```python
def is_palindrome(s):
    return s == s[::-1]
```

**Nova Pro** ✅

```python
def is_palindrome(s):
    return s == s[::-1]
```

**Llama 4 Scout** ✅

```python
def is_palindrome(s: str) -> bool:
    s = ''.join(c for c in s if c.isalnum()).lower()
    return s == s[::-1]

# Example usage:
if __name__ == "__main__":
    print(is_palindrome("A man, a plan, a canal: Panama"))  # True
    print(is_palindrome("Not a palindrome"))  # False
```


## Q4: List exactly three fruits, lowercase, separated by commas, with nothing else.

**Nova Micro**

apples, bananas, oranges

**Nova 2 Lite**

apple, banana, cherry

**Nova Pro**

apple, banana, orange

**Llama 4 Scout**

apple, banana, cherry


## Q5: Explain what an API is to a 10-year-old in two sentences.

**Nova Micro**

An API is like a set of instructions that lets different computer programs talk to each other and share information, kind of like how you might pass notes in class to get a message to a friend.

**Nova 2 Lite**

An API, short for Application Programming Interface, is like a special set of rules that lets different apps and websites talk to each other. It’s kind of like a menu at a restaurant—if you tell the app what you want (like looking up a weather forecast), the API is the helper that gets the information from the right place and brings it back to you.

**Nova Pro**

An API is like a special door that lets different computer programs talk to each other and share information. It helps them work together smoothly, like friends passing notes in class.

**Llama 4 Scout**

Imagine you have a lemonade stand and you want to share your lemonade with your friends who live in different houses. An API, or Application Programming Interface, is like a special messenger that helps your friends order lemonade from you without having to come to your stand, by sending a message to you with what they want and how much they want to pay.
