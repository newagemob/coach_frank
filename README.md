# Coach Frank
***an AI skateboarding coach***

<img src="https://github.com/newagemob/coach_frank/blob/main/images/coach_frank_headshot.png" width="200" height="200" alt="Coach Frank" />

---

Q: What is Coach Frank?

A: Coach Frank is an AI skateboarding coach. He uses a webcam to detect your skateboarding tricks and then gives you feedback on what you did well and what you can improve on.

---

Q: How do I use Coach Frank?

A: You can use Coach Frank by cloning this repo and [this one](https://github.com/LightningDrop/SkateboardML).

```bash
mkdir skateboard.ai
cd skateboard.ai
git clone https://github.com/newagemob/coach_frank.git
git clone https://github.com/LightningDrop/SkateboardML.git
```

Then run the `environment.yaml` file to create a conda environment with all the required dependencies.

```bash
conda env create -f environment.yaml
```

Then run the `trick_classifier.py` file to start training Coach Frank on your local machine. Be sure you've set your ENV variables for your local `Tricks` directory.

```bash
python trick_classifier.py
```
