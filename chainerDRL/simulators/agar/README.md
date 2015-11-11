# Agar.io Driver
A tool for automating interaction with Agar.io.

## Dependencies
`chromedriver` must be on your path. Download it [here](https://sites.google.com/a/chromium.org/chromedriver/)

[Burp Suite](https://portswigger.net/burp/) requires Java. This project has been tested with Java version 1.8.0_40. Java must also be on your path.

Then `pip install -r requirements.txt`. Ignore the warnings.

## Use
The driver uses Burp Suite to inject a few hooks into the agar client code to make automated interaction feasible. A copy of `burp` and its dependencies is included with this project.

First run:

```bash
cd burp

# Windows
.\suite.bat

# Mac/linux
bash ./suite.sh
```

Then run `agar_io_driver.py` as a standalone python executable or import it into your project.

## How it Works
`agar-io-driver.py` uses [Selenium](http://www.seleniumhq.org/) to automate launching a web browser and pointing it at agar.io. The script will start the game and move the mouse to different areas of the screen to move the player. Selenium will execute Javascript on the client to poll for the game's current state and to interact with the UI.

Selenium gives us almost everything we need, except for the score. The score is drawn to an HTML5 canvas by private Javascript functions embedded in a script tag in the HTML. AFAIK, there is no other way to access this score function without either creating our own Websockets connection to agar.io, or man-in-the-middling the browser's connection and injecting extra Javascript.

Burp allows automated proxying/man-in-the-middling of HTML traffic on your host. `burp/Lib/BurpExtender.py` creates a custom Burp extension that listens to all traffic and injects some HTML and Javascript for us.

On every HTML response Burp will add a `<div id="score"></div>` element at the end of the HTML body. It will also inject an extra line of Javascript to populate that div with the player's score on every iteration of the game loop.

## Security Concerns ("I don't trust you to man-in-the-middle my web traffic")
Burp will only intercept traffic that passes through its proxy at 127.0.0.1:8080. Your web browser will
not do this unless you explicitly configure it to.

`agar-io-driver.py` creates a new Chrome window configured to pass its traffic through the Burp proxy. This won't affect any other Chrome windows you use, past or present. Any new tabs on the Chrome window this script creates _will_ be man-in-the-middled though, so I guess don't use those tabs.

You can also take a look at `burp/Lib/BurpExtender.py` to see what's injected.

## Breaking Changes to Agar.io
The agar.io client code is obfuscated/minified. Getting a hook on the `get_score()` function requires a bit of reverse engineering. Fortunately this is easy -- just grep for "score" in the HTML source or the downloaded JavaScript and you'll find something like

```
P=Math.max(P,Db());0!=P&&(null==Ca&&(Ca=new Da(24,"#FFFFFF")),Ca.u(ga("score")+": "+~~(P/100)
```

Here, `P` is the player's best score so far (I think) and `Db()` is the `get_current_score()` function. The client draws whichever is greater to the score box in the canvas. We add

```
document.getElementById("score").innerHTML=Db();
```

right in the middle of that mess.

We also add a `<div id="score"></div>` element to the end of the body. This should continue to work if the Javascript is re-minified.

When agar.io updates it will almost certainly re-minify the client code and these variable names will change. Grep for `score` again, find the new variable/function names, and update them in `burp/Lib/BurpExtender.py`. Go ahead and make a PR here with the change too while you're at it.
