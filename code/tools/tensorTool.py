def subtract_(target, minuend, subtrahend):
  for name in target:
    target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()

def copy_weight(target, source):
  for name in target:
    target[name].data = source[name].data.clone()